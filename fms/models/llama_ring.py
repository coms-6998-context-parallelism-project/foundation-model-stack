from typing import Any, Optional, Tuple, Union
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.linalg
import torch.nn.functional as F
from fms.models.ring_attention_helper import RingAttentionHelper
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy

def _compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, layer_idx: int, tolerance: float = 1e-3, print_values: bool = False):
    print(f"[DEBUG L{layer_idx}] Comparing {name}...")
    sys.stdout.flush()
    if t1.shape != t2.shape:
        print(f"  MISMATCH SHAPE for {name}: Ring {t1.shape} vs NoRing {t2.shape}")
        sys.stdout.flush()
        return False
    t2c = t2.to(t1.dtype) if t1.dtype != t2.dtype else t2
    if not torch.allclose(t1, t2c, atol=tolerance, rtol=tolerance):
        diff = torch.abs(t1 - t2c)
        print(f"  MISMATCH VAL for {name} (MaxDiff: {diff.max().item():.2e}, MeanDiff: {diff.mean().item():.2e}, RingNorm: {torch.linalg.norm(t1.float()).item():.2e}, NoRingNorm: {torch.linalg.norm(t2c.float()).item():.2e})")
        sys.stdout.flush()
        if print_values:
            print(f"    Ring: {t1.flatten()[:5]}... {t1.flatten()[-5:]}")
            sys.stdout.flush()
            print(f"    NoRing: {t2c.flatten()[:5]}... {t2c.flatten()[-5:]}")
            sys.stdout.flush()
        return False
    print(f"  MATCH for {name} (RingNorm: {torch.linalg.norm(t1.float()).item():.2e}, NoRingNorm: {torch.linalg.norm(t2c.float()).item():.2e})")
    sys.stdout.flush()
    return True

def _compare_debug_data(debug_no_ring: dict, debug_ring: dict, strategy: RingAttentionStrategy, layer_idx: int, tolerance: float, print_values: bool, print_matches: bool, rank: int, prefix: str):
    print(f"\n--- Comparing Debug Data for Layer {layer_idx}, Rank {rank} ---")
    sys.stdout.flush()
    if not debug_no_ring or not debug_ring:
        print(f"[DEBUG] Missing data: NoRing={bool(debug_no_ring)}, Ring={bool(debug_ring)}")
        sys.stdout.flush()
        return
    valid_len = strategy.get_local_valid_len()
    total_len = getattr(strategy, '_original_seq_len', -1)
    print(f"[DEBUG L{layer_idx}] Rank {rank} Valid Len: {valid_len}, Total Len: {total_len}")
    sys.stdout.flush()
    comparison_map = [
        ("X_norm", "_x_norm_global", "_x_norm", 1, False),
        ("Q_local", "_q_global", "_attn_q_local", 2, False),
        ("K_local", "_k_global", "_attn_k_local", 2, False),
        ("V_local", "_v_global", "_attn_v_local", 2, False),
        ("SDP_Scores_K0", "_sdp_scores_kblock0_global", "_sdp_scores_kblock0", [2,3], False),
        ("SDP_Probs_K0", "_sdp_probs_kblock0_global", "_sdp_probs_kblock0", [2,3], False),
        ("Context_Raw", "_context_raw_global", "_context_raw", 2, False),
        ("Attn_Dense_Out", "_attn_out_dense_global", "_attn_out_dense", 1, False),
        ("Residual1", "_residual1_global", "_attn_out_residual", 1, False),
        ("FF_LN_Out", "_ff_ln_out_global", "_ff_ln_out", 1, False),
        ("FF_Out", "_ff_out_global", "_ff_out_raw", 1, False),
        ("Block_Output", "_block_output_global", "_block_output", 1, False),
        ("Mask_Sum", "_mask_slice_sumpass_kblock0_global_sum", "_sumpass_kblock0_mask_slice_sum", -1, False),
    ]
    all_match = True
    for name, gk, rk, dims, is_w in comparison_map:
        gkey, rkey = f"noring{gk}", f"{prefix}{rk}"
        if not is_w and gkey not in debug_no_ring:
            print(f"[DEBUG L{layer_idx}] Missing {gkey}")
            sys.stdout.flush(); all_match = False; continue
        if not is_w and rkey not in debug_ring:
            print(f"[DEBUG L{layer_idx}] Missing {rkey}")
            sys.stdout.flush(); all_match = False; continue
        t1 = debug_no_ring[gkey].to(torch.float32)
        t2 = debug_ring[rkey].to(torch.float32)
        if valid_len == 0:
            if t2.numel()==0: continue
            print(f"[DEBUG L{layer_idx}] MISMATCH {name}: empty vs non-empty")
            sys.stdout.flush(); all_match=False; continue
        sl = [slice(None)]*t1.ndim
        if not is_w:
            start = strategy.rank*strategy.block_size
            if isinstance(dims,int): sl[dims]=slice(start,start+valid_len)
            else: sl[dims[0]]=slice(start,start+valid_len); sl[dims[1]]=slice(0,valid_len)
            try: t1=t1[tuple(sl)]
            except: print(f"[DEBUG] Slice error {name}"); sys.stdout.flush(); all_match=False; continue
        if t1.shape!=t2.shape: print(f"[DEBUG L{layer_idx}] SHAPE MISMATCH {name}: {t2.shape} vs {t1.shape}"); sys.stdout.flush(); all_match=False; continue
        if not _compare_tensors(name,t2,t1,layer_idx,tolerance,print_values): all_match=False
    for k in (f"{prefix}_kahan_num_comp_norm",f"{prefix}_kahan_den_comp_norm"):
        if k in debug_ring:
            print(f"[DEBUG L{layer_idx}] {k}: {debug_ring[k].item():.3e}"); sys.stdout.flush()
    print(f"--- Result Layer {layer_idx}, Rank {rank}: {'ALL MATCH' if all_match else 'MISMATCHES'} ---")
    sys.stdout.flush()

def _perform_shadow_standard_attention_pass(self_block, x_norm, residual, mask, pos_ids, is_causal, attn_alg, block_size):
    print(f"[DEBUG L_shadow {self_block.layer_idx}] Shadow pass..."); sys.stdout.flush()
    debug_data={
        "noring_x_norm_global":x_norm.clone().cpu(),
        "noring_residual_global_input":residual.clone().cpu(),
        "noring_residual_input_global":residual.clone().cpu()
    }
    dbg = {}
    pk = "noring_shadow"
    out=self_block.attn(q=x_norm,k=x_norm,v=x_norm,mask=mask,position_ids=pos_ids,attn_algorithm=attn_alg,past_key_value_state=None,use_cache=False,is_self=True,is_causal_mask=is_causal,debug_dict=dbg,debug_key_prefix=pk)
    for suffix in ("q_final","k_final","v_final","sdp_scores_kblock0","sdp_probs_kblock0","context_raw","attn_out_dense"):
        key=f"{pk}_{suffix}"
        gkey=f"noring_{suffix.replace('_final','')}_global" if 'final' in suffix else f"noring_{suffix}_global"
        if key in dbg: debug_data[gkey]=dbg[key].clone().cpu() if 'dense' not in suffix else out.clone().cpu()
    if mask is not None and block_size>0:
        ql=min(block_size,x_norm.size(1)); kl=block_size
        sl=mask if mask.ndim<4 else mask[:, :, :ql, :kl]
        debug_data["noring_mask_slice_sumpass_kblock0_global_sum"]=sl.sum().cpu() if sl is not None and sl.numel()>0 else None
    r1=out+residual; debug_data["noring_residual1_global"]=r1.clone().cpu()
    ln=self_block.ff_ln(r1); debug_data["noring_ff_ln_out_global"]=ln.clone().cpu()
    ff = {}
    fp = "noring_shadow_ff"
    fo=self_block.ff_sub_layer(ln,debug_dict=ff,debug_key_prefix=fp); debug_data["noring_ff_out_global"]=fo.clone().cpu()
    debug_data["noring_block_output_global"]=(fo+r1).clone().cpu()
    print(f"[DEBUG L_shadow {self_block.layer_idx}] Completed {len(debug_data)} keys"); sys.stdout.flush()
    return debug_data

def forward_ring(self: nn.Module, x: torch.Tensor, *, mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache: bool=False, is_causal_mask: bool=False, attn_algorithm: Optional[str]=None, distributed_strategy: Optional[DistributedStrategy]=None, debug_dict_populate: Optional[dict]=None, debug_key_prefix_populate: str="", debug_print_values: bool=False, debug_tolerance: float=1e-3, layer_idx: int=-1) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
    x_out, cache_out, _ = self._forward_ring_attention(x,mask=mask,position_ids=position_ids,past_key_value_state=past_key_value_state,use_cache=use_cache,is_causal_mask=is_causal_mask,strategy=distributed_strategy,attn_algorithm=attn_algorithm,debug_dict_populate=debug_dict_populate,debug_key_prefix_populate=debug_key_prefix_populate,debug_print_values=debug_print_values,debug_tolerance=debug_tolerance,layer_idx=layer_idx)
    rank=distributed_strategy.rank if isinstance(distributed_strategy,RingAttentionStrategy) else 0
    if use_cache: return x_out, cache_out
    return x_out

def _forward_ring_attention(self: nn.Module, x: torch.Tensor, *, mask: Optional[torch.Tensor], position_ids: Optional[torch.Tensor], past_key_value_state: Optional[Tuple[torch.Tensor, torch.Tensor]], use_cache: bool, is_causal_mask: bool, attn_algorithm: Optional[str], strategy: DistributedStrategy, layer_idx: int=-1, debug_dict_populate: Optional[dict]=None, debug_print_matches: bool=False, debug_key_prefix_populate: str="", debug_print_values: bool=False, debug_tolerance: float=1e-3) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Any]:
    assert isinstance(strategy, RingAttentionStrategy)
    residual=x; x_norm=self.ln(x)
    rank= strategy.rank; valid_len=strategy.get_local_valid_len(); B,T,_=x.shape
    if position_ids is None and valid_len>0:
        start=rank*strategy.block_size
        pid=torch.arange(start,start+T,device=x.device).unsqueeze(0).expand(B,-1)
        pad= torch.arange(T,device=x.device).expand(B,-1) >= valid_len
        pid[pad]=-1; position_ids=pid
    helper=getattr(self,'ring_helper',None)
    if helper is None or helper.strategy is not strategy:
        self.ring_helper=RingAttentionHelper(attn_module=self.attn,strategy=strategy,llama_block=self,use_cache=use_cache,ff=self.ff_sub_layer,ff_norm=self.ff_ln)
    out, cache, _ = self.ring_helper.forward(x_norm,mask=mask,strategy=strategy,position_ids=position_ids,past_key_value_state=past_key_value_state,is_causal_mask=is_causal_mask,valid_len=valid_len,residual=residual,debug_dict_populate=debug_dict_populate,debug_key_prefix_populate=debug_key_prefix_populate,layer_idx=layer_idx,debug_print_values=debug_print_values,debug_tolerance=debug_tolerance)
    if self.config.debug_mode and layer_idx==self.config.debug_target_layer:
        nr_container = [None]
        # Gather shards
        shards = [torch.empty_like(x) for _ in range(strategy.world_size)]
        dist.all_gather(shards, x, group=strategy.group)
        # Gather position_ids shards
        template = position_ids if position_ids is not None else shards[0]
        psh = [torch.empty_like(template) for _ in range(strategy.world_size)]
        gather_src = position_ids if position_ids is not None else torch.full((B, T), -1, device=x.device)
        dist.all_gather(psh, gather_src, group=strategy.group)
        # On rank 0, reconstruct and compute shadow pass
        if rank == 0:
            orig_len = getattr(strategy, '_original_seq_len', None)
            xg = torch.cat(shards, dim=1)
            if orig_len is not None and xg.size(1) > orig_len:
                xg = xg[:, :orig_len, :]
            pg = torch.cat(psh, dim=1)
            if orig_len is not None and pg.size(1) > orig_len:
                pg = pg[:, :orig_len]
            self.eval()
            with torch.no_grad():
                xng = self.ln(xg)
                nr_container[0] = _perform_shadow_standard_attention_pass(
                    self, xng, xg, mask, pg, is_causal_mask, attn_algorithm, strategy.block_size
                )
            self.train()
        # Broadcast shadow data to all ranks
        dist.broadcast_object_list(nr_container, src=0, group=strategy.group)
        nr = nr_container[0]
        # Compare on target ranks
        if rank in [0, 1] and nr and debug_dict_populate:
            prefix = f"ring_r{rank}"
            _compare_debug_data(
                nr,
                debug_dict_populate,
                strategy,
                layer_idx,
                debug_tolerance,
                debug_print_values,
                debug_print_matches,
                rank,
                prefix,
            )
    return out, cache, None
