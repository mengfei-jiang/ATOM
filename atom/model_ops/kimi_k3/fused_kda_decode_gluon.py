# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused KDA decode kernel (Gluon, CDNA3/gfx950):
conv1d + delta-rule recurrence + gated RMSNorm.

Grid: (batch, heads) — one block per (sequence, head).
Phase 1: Conv1d Q + K + V done once, V stored to LDS.
Phase 2: Delta rule recurrence in V-chunks, V read from LDS.
Phase 3: Gated RMSNorm, raw output read from LDS.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _fused_kda_decode_gluon_kernel(
    # Conv1d
    x_ptr,  # [B, 3*lp] bf16, stride (stride_x_tok, 1) — may be slice of larger tensor
    conv_weight_ptr,  # [3, W, lp] fp32, width-major
    conv_state_ptr,  # [N, 3*lp, W-1] bf16, transposed view
    # Recurrence
    gate_ptr,  # [1, B, H, K] bf16, contiguous
    beta_ptr,  # [1, B, H] bf16, stride (., stride_beta_tok, 1) — may be slice
    A_log_ptr,  # [H] fp32
    dt_bias_ptr,  # [H*K] fp32
    # State
    ssm_state_ptr,  # [N, H, K, K] fp32 — note: fp32 for precision
    ssm_state_indices_ptr,  # [B] int32
    cu_seqlens_ptr,  # [B+1] int64
    # RMSNorm
    norm_weight_ptr,  # [K] fp32
    out_gate_ptr,  # [B, H, K] bf16, stride (stride_og_tok, K, 1) — may be slice
    # Output
    out_ptr,  # [B, H*K] bf16, contiguous
    # Scalars
    lower_bound,
    norm_eps,
    qk_scale,
    T: tl.int64,
    # Constexprs
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    W: gl.constexpr,
    CHUNK_V: gl.constexpr,
    N_CHUNKS: gl.constexpr,
    # Strides
    stride_x_tok: tl.int64,  # x row stride (may != 3*lp if sliced)
    stride_cw_group: tl.int64,  # conv_weight stride for Q/K/V group dim
    stride_cw_width: tl.int64,  # conv_weight stride for width dim
    stride_cw_ch: tl.int64,  # conv_weight stride for channel dim
    stride_cs_slot: tl.int64,
    stride_cs_dim: tl.int64,
    stride_cs_pos: tl.int64,
    stride_beta_tok: tl.int64,  # beta token stride (may != H)
    stride_og_tok: tl.int64,  # output_gate token stride (may != H*K)
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
    # In x: Q=[0:lp], K=[lp:2*lp], V=[2*lp:3*lp]
    q_ch_off = i_h * K  # Q channel offset within x
    k_ch_off = lp + i_h * K  # K channel offset within x
    v_ch_off = 2 * lp + i_h * V  # V channel offset within x

    # conv_weight indexing: base + channel * stride_cw_ch + width * stride_cw_width
    # 3D [3, W, lp]: stride_cw_group=W*lp, stride_cw_width=lp, stride_cw_ch=1
    # 2D [3*lp, W]:  stride_cw_group=lp*W, stride_cw_width=1,  stride_cw_ch=W
    cw_q_base = 0 * stride_cw_group + i_h * K * stride_cw_ch
    cw_k_base = 1 * stride_cw_group + i_h * K * stride_cw_ch
    cw_v_base = 2 * stride_cw_group + i_h * V * stride_cw_ch

    # --- Layouts ---
    k_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )
    o_k = gl.arange(0, K, layout=k_layout)

    cv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )

    hk_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    hk_v_slice: gl.constexpr = gl.SliceLayout(1, hk_layout)
    hk_k_slice: gl.constexpr = gl.SliceLayout(0, hk_layout)

    # --- Shared memory ---
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    smem_v = gl.allocate_shared_memory(tl.float32, [V], smem_layout)
    smem_out = gl.allocate_shared_memory(tl.float32, [V], smem_layout)

    o_v_full = gl.arange(0, V, layout=k_layout)

    for i_t in range(seq_T):
        tok = bos + i_t

        p_x = x_ptr + tok * stride_x_tok
        p_cs = conv_state_ptr + state_idx * stride_cs_slot

        # ============================================================
        # Phase 1: Conv1d Q + K + V (full head_dim, done once)
        # ============================================================

        # Q conv1d
        b_x_q = gl.load(p_x + q_ch_off + o_k).to(tl.float32)
        p_csq = p_cs + (q_ch_off + o_k) * stride_cs_dim
        b_q = b_x_q * gl.load(
            conv_weight_ptr + cw_q_base + o_k * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        for j in gl.static_range(W - 1):
            cs = gl.load(p_csq + j * stride_cs_pos).to(tl.float32)
            cw = gl.load(
                conv_weight_ptr + cw_q_base + o_k * stride_cw_ch + j * stride_cw_width
            ).to(tl.float32)
            b_q = b_q + cs * cw
        b_q = b_q * tl.sigmoid(b_q)
        for j in gl.static_range(W - 2):
            src = gl.load(p_csq + (j + 1) * stride_cs_pos)
            gl.store(p_csq + j * stride_cs_pos, src)
        gl.store(p_csq + (W - 2) * stride_cs_pos, b_x_q.to(p_csq.dtype.element_ty))

        # K conv1d
        b_x_k = gl.load(p_x + k_ch_off + o_k).to(tl.float32)
        p_csk = p_cs + (k_ch_off + o_k) * stride_cs_dim
        b_k = b_x_k * gl.load(
            conv_weight_ptr + cw_k_base + o_k * stride_cw_ch + (W - 1) * stride_cw_width
        ).to(tl.float32)
        for j in gl.static_range(W - 1):
            cs = gl.load(p_csk + j * stride_cs_pos).to(tl.float32)
            cw = gl.load(
                conv_weight_ptr + cw_k_base + o_k * stride_cw_ch + j * stride_cw_width
            ).to(tl.float32)
            b_k = b_k + cs * cw
        b_k = b_k * tl.sigmoid(b_k)
        for j in gl.static_range(W - 2):
            src = gl.load(p_csk + (j + 1) * stride_cs_pos)
            gl.store(p_csk + j * stride_cs_pos, src)
        gl.store(p_csk + (W - 2) * stride_cs_pos, b_x_k.to(p_csk.dtype.element_ty))

        # V conv1d (full V, store to LDS)
        b_x_v = gl.load(p_x + v_ch_off + o_v_full).to(tl.float32)
        p_csv = p_cs + (v_ch_off + o_v_full) * stride_cs_dim
        b_v_full = b_x_v * gl.load(
            conv_weight_ptr
            + cw_v_base
            + o_v_full * stride_cw_ch
            + (W - 1) * stride_cw_width
        ).to(tl.float32)
        for j in gl.static_range(W - 1):
            cs = gl.load(p_csv + j * stride_cs_pos).to(tl.float32)
            cw = gl.load(
                conv_weight_ptr
                + cw_v_base
                + o_v_full * stride_cw_ch
                + j * stride_cw_width
            ).to(tl.float32)
            b_v_full = b_v_full + cs * cw
        b_v_full = b_v_full * tl.sigmoid(b_v_full)
        for j in gl.static_range(W - 2):
            src = gl.load(p_csv + (j + 1) * stride_cs_pos)
            gl.store(p_csv + j * stride_cs_pos, src)
        gl.store(p_csv + (W - 2) * stride_cs_pos, b_x_v.to(p_csv.dtype.element_ty))
        smem_v.store(b_v_full)

        # QK L2 Norm
        b_q = b_q * tl.math.rsqrt(gl.sum(b_q * b_q) + 1e-6) * qk_scale
        b_k = b_k * tl.math.rsqrt(gl.sum(b_k * b_k) + 1e-6)

        # Decay gate: gate is [1, B, H, K] contiguous
        b_a = gl.load(gate_ptr + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_dt = gl.load(dt_bias_ptr + i_h * K + o_k).to(tl.float32)
        b_A = gl.load(A_log_ptr + i_h).to(tl.float32)
        b_g = lower_bound * tl.sigmoid(tl.exp(b_A) * (b_a + b_dt))

        # Beta: [1, B, H] with stride (., stride_beta_tok, 1)
        b_beta = tl.sigmoid(
            gl.load(beta_ptr + tok * stride_beta_tok + i_h).to(tl.float32)
        )

        # ============================================================
        # Phase 2: Delta Rule (V in chunks, V read from LDS)
        # ============================================================
        o_sumsq = 0.0
        p_state_base = ssm_state_ptr + state_idx * stride_ssm_slot + i_h * V * K

        for i_c in gl.static_range(N_CHUNKS):
            o_v = i_c * CHUNK_V + gl.arange(0, CHUNK_V, layout=cv_layout)
            mask_v = o_v < V

            b_v = smem_v.slice(i_c * CHUNK_V, CHUNK_V).load(layout=cv_layout)

            # Load state [CHUNK_V, K] — fp32
            o_v_2d = gl.arange(0, CHUNK_V, layout=gl.SliceLayout(1, hk_layout))
            o_k_2d = gl.arange(0, K, layout=gl.SliceLayout(0, hk_layout))
            mask_h = (o_v_2d[:, None] < V) & (o_k_2d[None, :] < K)
            p_h = p_state_base + (i_c * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
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

            # Store state back — fp32
            gl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)

            o_sumsq = o_sumsq + gl.sum(b_o_cv * b_o_cv)
            smem_out.slice(i_c * CHUNK_V, CHUNK_V).store(b_o_cv)

        # ============================================================
        # Phase 3: Gated RMSNorm (read raw output from LDS)
        # ============================================================
        rstd = tl.math.rsqrt(o_sumsq / V + norm_eps)

        for i_c in gl.static_range(N_CHUNKS):
            o_v = i_c * CHUNK_V + gl.arange(0, CHUNK_V, layout=cv_layout)
            mask_v = o_v < V
            b_raw = smem_out.slice(i_c * CHUNK_V, CHUNK_V).load(layout=cv_layout)
            # norm_weight is fp32
            b_w = gl.load(norm_weight_ptr + o_v, mask=mask_v, other=0.0).to(tl.float32)
            # output_gate: [B, H, K] stride (stride_og_tok, K, 1)
            b_og = gl.load(
                out_gate_ptr + tok * stride_og_tok + i_h * V + o_v,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
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
    W = conv_weight.shape[-2] if conv_weight.dim() == 3 else conv_weight.shape[-1]
    lp = H * K
    CHUNK_V = min(triton.next_power_of_2(V), 32)
    N_CHUNKS = triton.cdiv(V, CHUNK_V)
    batch = cu_seqlens.shape[0] - 1
    out = torch.empty(T, lp, dtype=torch.bfloat16, device=mixed_qkv.device)

    # Handle conv_weight shape: [3, W, lp] or [3*lp, W]
    if conv_weight.dim() == 3:
        # [3, W, lp]: group=stride(0), width=stride(1), ch=stride(2)=1
        stride_cw_group = conv_weight.stride(0)
        stride_cw_width = conv_weight.stride(1)
        stride_cw_ch = conv_weight.stride(2)
    else:
        # [3*lp, W]: group=lp*stride(0), width=stride(1), ch=stride(0)
        stride_cw_group = lp * conv_weight.stride(0)
        stride_cw_width = conv_weight.stride(1)
        stride_cw_ch = conv_weight.stride(0)

    # Handle beta stride
    if beta.dim() == 3:
        stride_beta_tok = beta.stride(1)
    else:
        stride_beta_tok = beta.stride(0)

    # Handle output_gate stride
    stride_og_tok = out_gate.stride(0)

    grid = (batch, H)
    _fused_kda_decode_gluon_kernel[grid](
        mixed_qkv,
        conv_weight,
        conv_state,
        gate,
        beta,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_indices,
        cu_seqlens,
        norm_weight,
        out_gate,
        out,
        lower_bound,
        norm_eps,
        K**-0.5,
        T,
        H=H,
        K=K,
        V=V,
        W=W,
        CHUNK_V=CHUNK_V,
        N_CHUNKS=N_CHUNKS,
        stride_x_tok=mixed_qkv.stride(0),
        stride_cw_group=stride_cw_group,
        stride_cw_width=stride_cw_width,
        stride_cw_ch=stride_cw_ch,
        stride_cs_slot=conv_state.stride(0),
        stride_cs_dim=conv_state.stride(1),
        stride_cs_pos=conv_state.stride(2),
        stride_beta_tok=stride_beta_tok,
        stride_og_tok=stride_og_tok,
        stride_ssm_slot=ssm_state.stride(0),
        num_warps=4,
    )
    return out
