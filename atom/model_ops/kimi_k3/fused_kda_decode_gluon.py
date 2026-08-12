# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused KDA decode kernel (Gluon, CDNA3/gfx950):
conv1d + delta-rule recurrence + gated RMSNorm.

Grid: (batch, heads) — one block per (sequence, head).
Phase 1: Conv1d Q + K + V done once for the full head_dim.
Phase 2: Delta rule recurrence in V-chunks, V conv result read from LDS.
Phase 3: Gated RMSNorm, raw output read from LDS.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _fused_kda_decode_gluon_kernel(
    mixed_qkv_ptr, conv_weight_ptr, conv_state_ptr,
    gate_ptr, beta_ptr, A_log_ptr, dt_bias_ptr,
    ssm_state_ptr, ssm_state_indices_ptr, cu_seqlens_ptr,
    norm_weight_ptr, out_gate_ptr, out_ptr,
    lower_bound, norm_eps, qk_scale,
    T: tl.int64,
    H: gl.constexpr, K: gl.constexpr, V: gl.constexpr,
    W: gl.constexpr, CHUNK_V: gl.constexpr, N_CHUNKS: gl.constexpr,
    stride_mqkv_tok: tl.int64, stride_cw_dim: tl.int64,
    stride_cs_slot: tl.int64, stride_cs_dim: tl.int64, stride_cs_pos: tl.int64,
    stride_ssm_slot: tl.int64,
):
    i_n = gl.program_id(0)
    i_h = gl.program_id(1)

    bos = gl.load(cu_seqlens_ptr + i_n).to(tl.int64)
    eos = gl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
    seq_T = eos - bos
    if seq_T == 0:
        return

    state_idx = gl.load(ssm_state_indices_ptr + i_n).to(tl.int64)
    if state_idx < 0:
        return

    lp = H * K
    q_base = i_h * K
    k_base = lp + i_h * K
    v_base = 2 * lp + i_h * V

    # --- Layouts ---
    # 1D layout for K-sized or V-sized vectors (capacity 256 ≥ 128)
    k_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[64],
        warps_per_cta=[4], order=[0],
    )
    o_k = gl.arange(0, K, layout=k_layout)

    cv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1], threads_per_warp=[64],
        warps_per_cta=[4], order=[0],
    )

    # 2D layout for state [CHUNK_V=32, K=128]
    hk_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    hk_v_slice: gl.constexpr = gl.SliceLayout(1, hk_layout)
    hk_k_slice: gl.constexpr = gl.SliceLayout(0, hk_layout)

    # --- Shared memory ---
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    smem_v = gl.allocate_shared_memory(tl.float32, [V], smem_layout)
    smem_out = gl.allocate_shared_memory(tl.float32, [V], smem_layout)

    # V uses same layout as K for the full-dim conv1d pass
    o_v_full = gl.arange(0, V, layout=k_layout)

    for i_t in range(seq_T):
        tok = bos + i_t
        p_mqkv = mixed_qkv_ptr + tok * stride_mqkv_tok
        p_cs = conv_state_ptr + state_idx * stride_cs_slot

        # ============================================================
        # Phase 1: Conv1d Q + K + V (full head_dim, done once)
        # ============================================================

        # --- Q conv1d ---
        p_xq = p_mqkv + q_base + o_k
        b_x_q = gl.load(p_xq).to(tl.float32)
        p_csq = p_cs + (q_base + o_k) * stride_cs_dim
        b_q = b_x_q * gl.load(conv_weight_ptr + (q_base + o_k) * stride_cw_dim + (W - 1)).to(tl.float32)
        for j in gl.static_range(W - 1):
            cs = gl.load(p_csq + j * stride_cs_pos).to(tl.float32)
            cw = gl.load(conv_weight_ptr + (q_base + o_k) * stride_cw_dim + j).to(tl.float32)
            b_q = b_q + cs * cw
        b_q = b_q * tl.sigmoid(b_q)
        for j in gl.static_range(W - 2):
            src = gl.load(p_csq + (j + 1) * stride_cs_pos)
            gl.store(p_csq + j * stride_cs_pos, src)
        gl.store(p_csq + (W - 2) * stride_cs_pos, b_x_q.to(p_csq.dtype.element_ty))

        # --- K conv1d ---
        p_xk = p_mqkv + k_base + o_k
        b_x_k = gl.load(p_xk).to(tl.float32)
        p_csk = p_cs + (k_base + o_k) * stride_cs_dim
        b_k = b_x_k * gl.load(conv_weight_ptr + (k_base + o_k) * stride_cw_dim + (W - 1)).to(tl.float32)
        for j in gl.static_range(W - 1):
            cs = gl.load(p_csk + j * stride_cs_pos).to(tl.float32)
            cw = gl.load(conv_weight_ptr + (k_base + o_k) * stride_cw_dim + j).to(tl.float32)
            b_k = b_k + cs * cw
        b_k = b_k * tl.sigmoid(b_k)
        for j in gl.static_range(W - 2):
            src = gl.load(p_csk + (j + 1) * stride_cs_pos)
            gl.store(p_csk + j * stride_cs_pos, src)
        gl.store(p_csk + (W - 2) * stride_cs_pos, b_x_k.to(p_csk.dtype.element_ty))

        # --- V conv1d (full V dim, store to LDS) ---
        p_xv = p_mqkv + v_base + o_v_full
        b_x_v = gl.load(p_xv).to(tl.float32)
        p_csv = p_cs + (v_base + o_v_full) * stride_cs_dim
        b_v_full = b_x_v * gl.load(
            conv_weight_ptr + (v_base + o_v_full) * stride_cw_dim + (W - 1)).to(tl.float32)
        for j in gl.static_range(W - 1):
            cs = gl.load(p_csv + j * stride_cs_pos).to(tl.float32)
            cw = gl.load(conv_weight_ptr + (v_base + o_v_full) * stride_cw_dim + j).to(tl.float32)
            b_v_full = b_v_full + cs * cw
        b_v_full = b_v_full * tl.sigmoid(b_v_full)
        for j in gl.static_range(W - 2):
            src = gl.load(p_csv + (j + 1) * stride_cs_pos)
            gl.store(p_csv + j * stride_cs_pos, src)
        gl.store(p_csv + (W - 2) * stride_cs_pos, b_x_v.to(p_csv.dtype.element_ty))

        # Store V conv result to LDS (read per-chunk in Phase 2)
        smem_v.store(b_v_full)

        # --- QK L2 Norm ---
        b_q = b_q * tl.math.rsqrt(gl.sum(b_q * b_q) + 1e-6) * qk_scale
        b_k = b_k * tl.math.rsqrt(gl.sum(b_k * b_k) + 1e-6)

        # --- Decay gate ---
        b_a = gl.load(gate_ptr + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_dt = gl.load(dt_bias_ptr + i_h * K + o_k).to(tl.float32)
        b_A = gl.load(A_log_ptr + i_h).to(tl.float32)
        b_g = lower_bound * tl.sigmoid(tl.exp(b_A) * (b_a + b_dt))

        # --- Beta ---
        b_beta = tl.sigmoid(gl.load(beta_ptr + tok * H + i_h).to(tl.float32))

        # ============================================================
        # Phase 2: Delta Rule (V in chunks, V conv read from LDS)
        # ============================================================
        o_sumsq = 0.0

        for i_c in gl.static_range(N_CHUNKS):
            o_v = i_c * CHUNK_V + gl.arange(0, CHUNK_V, layout=cv_layout)
            mask_v = o_v < V

            # Read V conv result from LDS (done once in Phase 1)
            smem_v_chunk = smem_v.slice(i_c * CHUNK_V, CHUNK_V)
            b_v = smem_v_chunk.load(layout=cv_layout)

            # Load state [CHUNK_V, K]
            o_v_2d = gl.arange(0, CHUNK_V, layout=gl.SliceLayout(1, hk_layout))
            o_k_2d = gl.arange(0, K, layout=gl.SliceLayout(0, hk_layout))
            mask_h = (o_v_2d[:, None] < V) & (o_k_2d[None, :] < K)

            p_h = (ssm_state_ptr + state_idx * stride_ssm_slot
                   + i_h * V * K
                   + (i_c * CHUNK_V + o_v_2d)[:, None] * K
                   + o_k_2d[None, :])
            b_h = gl.load(p_h, mask=mask_h, other=0.0).to(tl.float32)

            # Decay
            b_g_2d = gl.convert_layout(b_g, layout=gl.SliceLayout(0, hk_layout))
            b_h = b_h * tl.exp(b_g_2d[None, :])

            # Delta rule
            b_k_2d = gl.convert_layout(b_k, layout=hk_k_slice)
            b_dot = gl.sum(b_h * b_k_2d[None, :], axis=1)
            b_dot_cv = gl.convert_layout(b_dot, layout=cv_layout)
            b_v = b_v - b_dot_cv
            b_v = b_v * b_beta
            b_v_2d = gl.convert_layout(b_v, layout=hk_v_slice)
            b_h = b_h + b_v_2d[:, None] * b_k_2d[None, :]

            # Query output
            b_q_2d = gl.convert_layout(b_q, layout=hk_k_slice)
            b_o = gl.sum(b_h * b_q_2d[None, :], axis=1)
            b_o_cv = gl.convert_layout(b_o, layout=cv_layout)

            # Store state
            gl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)

            # Accumulate sumsq
            o_sumsq = o_sumsq + gl.sum(b_o_cv * b_o_cv)

            # Store raw output to LDS
            smem_out_chunk = smem_out.slice(i_c * CHUNK_V, CHUNK_V)
            smem_out_chunk.store(b_o_cv)

        # ============================================================
        # Phase 3: Gated RMSNorm (read raw output from LDS)
        # ============================================================
        rstd = tl.math.rsqrt(o_sumsq / V + norm_eps)

        for i_c in gl.static_range(N_CHUNKS):
            o_v = i_c * CHUNK_V + gl.arange(0, CHUNK_V, layout=cv_layout)
            mask_v = o_v < V

            smem_out_chunk = smem_out.slice(i_c * CHUNK_V, CHUNK_V)
            b_raw = smem_out_chunk.load(layout=cv_layout)
            b_w = gl.load(norm_weight_ptr + o_v, mask=mask_v, other=0.0).to(tl.float32)
            b_og = gl.load(out_gate_ptr + tok * (H * V) + i_h * V + o_v,
                           mask=mask_v, other=0.0).to(tl.float32)
            b_y = b_raw * rstd * b_w * tl.sigmoid(b_og)

            p_out_c = out_ptr + tok * (H * V) + i_h * V + o_v
            gl.store(p_out_c, b_y.to(out_ptr.dtype.element_ty), mask=mask_v)


def fused_kda_decode_gluon(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    out_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    head_dim: int,
    num_local_heads: int,
    lower_bound: float,
) -> torch.Tensor:
    """Fused KDA decode: conv1d + recurrence + gated RMSNorm."""
    T = mixed_qkv.shape[0]
    K = V = head_dim
    H = num_local_heads
    W = conv_weight.shape[-1]
    lp = H * K
    CHUNK_V = min(triton.next_power_of_2(V), 32)
    N_CHUNKS = triton.cdiv(V, CHUNK_V)
    batch = cu_seqlens.shape[0] - 1
    out = torch.empty(T, lp, dtype=torch.bfloat16, device=mixed_qkv.device)

    grid = (batch, H)
    _fused_kda_decode_gluon_kernel[grid](
        mixed_qkv, conv_weight, conv_state,
        gate, beta, A_log, dt_bias,
        ssm_state, ssm_state_indices, cu_seqlens,
        norm_weight, out_gate, out,
        lower_bound, norm_eps, K**-0.5, T,
        H=H, K=K, V=V, W=W, CHUNK_V=CHUNK_V, N_CHUNKS=N_CHUNKS,
        stride_mqkv_tok=mixed_qkv.stride(0),
        stride_cw_dim=conv_weight.stride(0),
        stride_cs_slot=conv_state.stride(0),
        stride_cs_dim=conv_state.stride(1),
        stride_cs_pos=conv_state.stride(2),
        stride_ssm_slot=ssm_state.stride(0),
        num_warps=4,
    )
    return out
