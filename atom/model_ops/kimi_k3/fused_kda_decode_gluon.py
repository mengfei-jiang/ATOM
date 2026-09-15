# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused KDA decode kernel (Gluon, CDNA3/gfx950):
conv1d + delta-rule recurrence + gated RMSNorm.

Grid: (batch, heads) — one block per (sequence, head).
Phase 1: Conv1d Q + K + V done once, V stored to LDS.
Phase 2: Delta rule recurrence in V-chunks, V read from LDS.
Phase 3: Gated RMSNorm, raw output read from LDS.

Four modes controlled by two constexpr flags:

  USE_REPLAY=False, IS_SPEC=False  →  normal decode (state from ssm_state)
  USE_REPLAY=False, IS_SPEC=True   →  DSpark spec decode (2D indices, snapshot)
  USE_REPLAY=True,  IS_SPEC=False  →  ReplaySSM (checkpoint rebuild + records)
  USE_REPLAY=True,  IS_SPEC=True   →  DSpark + ReplaySSM
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _fused_kda_decode_gluon_kernel(
    # Conv1d
    x_ptr,
    conv_weight_ptr,
    conv_state_ptr,
    # Recurrence
    gate_ptr,
    beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    # State
    ssm_state_ptr,
    ssm_state_indices_ptr,
    cu_seqlens_ptr,
    # RMSNorm + output
    norm_weight_ptr,
    out_gate_ptr,
    out_ptr,
    # Spec decode
    num_accepted_tokens_ptr,
    state_indices_ptr,
    conv_state_indices_ptr,
    # ReplaySSM
    ckpt_ptr,
    buf_k_ptr,
    buf_u_ptr,
    buf_g_ptr,
    write_pos_ptr,
    slot_idx_ptr,
    # Scalars
    lower_bound,
    norm_eps,
    qk_scale,
    T_tot: tl.int64,
    # Constexprs — dimensions
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    W: gl.constexpr,
    CHUNK_V: gl.constexpr,
    N_CHUNKS: gl.constexpr,
    STATE_LEN: gl.constexpr,
    # Constexprs — replay
    CAP: gl.constexpr,
    BH: gl.constexpr,
    # Constexprs — mode flags
    USE_REPLAY: gl.constexpr,
    IS_SPEC: gl.constexpr,
    # Strides — conv
    stride_x_tok: tl.int64,
    stride_cw_group: tl.int64,
    stride_cw_width: tl.int64,
    stride_cw_ch: tl.int64,
    stride_cs_slot: tl.int64,
    stride_cs_dim: tl.int64,
    stride_cs_pos: tl.int64,
    # Strides — recurrence
    stride_beta_tok: tl.int64,
    stride_og_tok: tl.int64,
    # Strides — state
    stride_ssm_slot: tl.int64,
    stride_si_seq: tl.int64,
    stride_si_tok: tl.int64,
    # Strides — checkpoint
    stride_ckpt_slot: tl.int64,
    # Strides — ring buffers
    stride_bufk_slot: tl.int64,
    stride_bufk_hv: tl.int64,
    stride_bufk_pos: tl.int64,
    stride_bufu_slot: tl.int64,
    stride_bufu_hv: tl.int64,
    stride_bufu_pos: tl.int64,
    stride_bufg_slot: tl.int64,
    stride_bufg_hv: tl.int64,
    stride_bufg_pos: tl.int64,
):
    i_n = gl.program_id(0)
    i_h = gl.program_id(1)

    bos = gl.load(cu_seqlens_ptr + i_n).to(tl.int64)
    eos = gl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
    seq_T = eos - bos
    if seq_T == 0:
        return

    if IS_SPEC:
        i_t_start = gl.load(num_accepted_tokens_ptr + i_n).to(tl.int64)
        i_t_start = tl.maximum(i_t_start - 1, i_t_start - i_t_start)
    else:
        i_t_start = bos - bos

    # ================================================================
    # Resolve slot + load initial state
    # ================================================================
    if USE_REPLAY:
        slot = gl.load(slot_idx_ptr + i_n).to(tl.int64)
        if slot < 0:
            return
        if IS_SPEC:
            conv_slot = gl.load(conv_state_indices_ptr + i_n).to(tl.int64)
        else:
            conv_slot = slot

        h_cursor = gl.load(write_pos_ptr + slot).to(tl.int32)
        h_cursor = tl.maximum(h_cursor, h_cursor - h_cursor)
        do_flush = h_cursor + 2 * seq_T > CAP
        do_flush_i64 = do_flush.to(tl.int64)
        base = h_cursor.to(tl.int64) * (1 - do_flush_i64)

        state_idx = slot
    else:
        if IS_SPEC:
            state_idx = gl.load(
                state_indices_ptr + i_n * stride_si_seq + i_t_start * stride_si_tok
            ).to(tl.int64)
            conv_slot = gl.load(conv_state_indices_ptr + i_n).to(tl.int64)
        else:
            state_idx = gl.load(ssm_state_indices_ptr + i_n).to(tl.int64)
            conv_slot = state_idx
        if state_idx < 0:
            return
        base = bos - bos

    lp = H * K
    q_ch_off = i_h * K
    k_ch_off = lp + i_h * K
    v_ch_off = 2 * lp + i_h * V

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

    p_cs = conv_state_ptr + conv_slot * stride_cs_slot

    # ================================================================
    # USE_REPLAY: checkpoint rebuild into ssm_state working buffer
    # Precompute kw = k * exp(suffix_decay) once (cumsum math).
    # Only one BH*K smem buffer (kw). g re-loaded from global in pass 2
    # to halve LDS footprint → better occupancy.
    # ================================================================
    if USE_REPLAY:
        smem_gk_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
        smem_kw_hist = gl.allocate_shared_memory(tl.float32, [BH * K], smem_gk_layout)
        smem_exp_ctot = gl.allocate_shared_memory(tl.float32, [K], smem_gk_layout)
        if h_cursor > 0:
            # Pass 1: load k to smem, accumulate ctot from g (g stays in global)
            b_ctot = (o_k - o_k).to(tl.float32)
            for h_i in gl.static_range(BH):
                if h_i < h_cursor:
                    g_val = gl.load(
                        buf_g_ptr + slot * stride_bufg_slot
                        + i_h * stride_bufg_hv + h_i * stride_bufg_pos + o_k,
                    ).to(tl.float32)
                    k_val = gl.load(
                        buf_k_ptr + slot * stride_bufk_slot
                        + i_h * stride_bufk_hv + h_i * stride_bufk_pos + o_k,
                    ).to(tl.float32)
                    b_ctot = b_ctot + g_val
                    smem_kw_hist.slice(h_i * K, K).store(k_val)

            # Pass 2: re-load g from global, compute kw = k * exp(suffix_decay)
            b_prefix = (o_k - o_k).to(tl.float32)
            for h_i in gl.static_range(BH):
                if h_i < h_cursor:
                    g_val = gl.load(
                        buf_g_ptr + slot * stride_bufg_slot
                        + i_h * stride_bufg_hv + h_i * stride_bufg_pos + o_k,
                    ).to(tl.float32)
                    k_val = smem_kw_hist.slice(h_i * K, K).load(layout=k_layout)
                    b_prefix = b_prefix + g_val
                    kw_val = k_val * tl.exp(b_ctot - b_prefix)
                    smem_kw_hist.slice(h_i * K, K).store(kw_val)

            smem_exp_ctot.slice(0, K).store(tl.exp(b_ctot))

        # Replay V-chunks. When N_CHUNKS==4: load all chunks into
        # registers, swap loop order (h_i outer) so kw is loaded once
        # per replay step → 4× fewer smem loads, all ckpt loads in
        # flight simultaneously.
        o_v_2d = gl.arange(0, CHUNK_V, layout=gl.SliceLayout(1, hk_layout))
        o_k_2d = gl.arange(0, K, layout=gl.SliceLayout(0, hk_layout))
        o_v_hk = gl.arange(0, CHUNK_V, layout=hk_v_slice)

        if N_CHUNKS == 4:
            p_ckpt_base = ckpt_ptr + slot * stride_ckpt_slot + i_h * V * K
            p_ssm_base = ssm_state_ptr + state_idx * stride_ssm_slot + i_h * V * K

            mask_0 = ((0 * CHUNK_V + o_v_2d)[:, None] < V) & (o_k_2d[None, :] < K)
            mask_1 = ((1 * CHUNK_V + o_v_2d)[:, None] < V) & (o_k_2d[None, :] < K)
            mask_2 = ((2 * CHUNK_V + o_v_2d)[:, None] < V) & (o_k_2d[None, :] < K)
            mask_3 = ((3 * CHUNK_V + o_v_2d)[:, None] < V) & (o_k_2d[None, :] < K)

            p_ck0 = p_ckpt_base + (0 * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
            p_ck1 = p_ckpt_base + (1 * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
            p_ck2 = p_ckpt_base + (2 * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
            p_ck3 = p_ckpt_base + (3 * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]

            b_h0 = gl.load(p_ck0, mask=mask_0, other=0.0).to(tl.float32)
            b_h1 = gl.load(p_ck1, mask=mask_1, other=0.0).to(tl.float32)
            b_h2 = gl.load(p_ck2, mask=mask_2, other=0.0).to(tl.float32)
            b_h3 = gl.load(p_ck3, mask=mask_3, other=0.0).to(tl.float32)

            if h_cursor > 0:
                b_decay = smem_exp_ctot.slice(0, K).load(layout=hk_k_slice)
                b_h0 = b_h0 * b_decay[None, :]
                b_h1 = b_h1 * b_decay[None, :]
                b_h2 = b_h2 * b_decay[None, :]
                b_h3 = b_h3 * b_decay[None, :]

                p_u_base = (buf_u_ptr + slot * stride_bufu_slot
                            + i_h * stride_bufu_hv)
                for h_i in gl.static_range(BH):
                    if h_i < h_cursor:
                        b_kw = smem_kw_hist.slice(h_i * K, K).load(layout=hk_k_slice)
                        p_u_h = p_u_base + h_i * stride_bufu_pos
                        b_u0 = gl.load(p_u_h + (0 * CHUNK_V + o_v_hk),
                                       mask=o_v_hk < CHUNK_V, other=0.0).to(tl.float32)
                        b_u1 = gl.load(p_u_h + (1 * CHUNK_V + o_v_hk),
                                       mask=o_v_hk < CHUNK_V, other=0.0).to(tl.float32)
                        b_u2 = gl.load(p_u_h + (2 * CHUNK_V + o_v_hk),
                                       mask=o_v_hk < CHUNK_V, other=0.0).to(tl.float32)
                        b_u3 = gl.load(p_u_h + (3 * CHUNK_V + o_v_hk),
                                       mask=o_v_hk < CHUNK_V, other=0.0).to(tl.float32)
                        b_h0 = b_h0 + b_u0[:, None] * b_kw[None, :]
                        b_h1 = b_h1 + b_u1[:, None] * b_kw[None, :]
                        b_h2 = b_h2 + b_u2[:, None] * b_kw[None, :]
                        b_h3 = b_h3 + b_u3[:, None] * b_kw[None, :]

            if do_flush:
                gl.store(p_ck0, b_h0.to(ckpt_ptr.dtype.element_ty), mask=mask_0)
                gl.store(p_ck1, b_h1.to(ckpt_ptr.dtype.element_ty), mask=mask_1)
                gl.store(p_ck2, b_h2.to(ckpt_ptr.dtype.element_ty), mask=mask_2)
                gl.store(p_ck3, b_h3.to(ckpt_ptr.dtype.element_ty), mask=mask_3)

            p_s0 = p_ssm_base + (0 * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
            p_s1 = p_ssm_base + (1 * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
            p_s2 = p_ssm_base + (2 * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
            p_s3 = p_ssm_base + (3 * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
            gl.store(p_s0, b_h0.to(ssm_state_ptr.dtype.element_ty), mask=mask_0)
            gl.store(p_s1, b_h1.to(ssm_state_ptr.dtype.element_ty), mask=mask_1)
            gl.store(p_s2, b_h2.to(ssm_state_ptr.dtype.element_ty), mask=mask_2)
            gl.store(p_s3, b_h3.to(ssm_state_ptr.dtype.element_ty), mask=mask_3)

        else:
            for i_c in gl.static_range(N_CHUNKS):
                mask_h = ((i_c * CHUNK_V + o_v_2d)[:, None] < V) & (o_k_2d[None, :] < K)
                p_ckpt = (
                    ckpt_ptr + slot * stride_ckpt_slot + i_h * V * K
                    + (i_c * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
                )
                b_h = gl.load(p_ckpt, mask=mask_h, other=0.0).to(tl.float32)

                if h_cursor > 0:
                    b_decay = smem_exp_ctot.slice(0, K).load(layout=hk_k_slice)
                    b_h = b_h * b_decay[None, :]
                    for h_i in gl.static_range(BH):
                        if h_i < h_cursor:
                            b_kw = smem_kw_hist.slice(h_i * K, K).load(layout=hk_k_slice)
                            b_u_h = gl.load(
                                buf_u_ptr + slot * stride_bufu_slot
                                + i_h * stride_bufu_hv + h_i * stride_bufu_pos
                                + (i_c * CHUNK_V + o_v_hk),
                                mask=o_v_hk < CHUNK_V, other=0.0,
                            ).to(tl.float32)
                            b_h = b_h + b_u_h[:, None] * b_kw[None, :]

                if do_flush:
                    gl.store(p_ckpt, b_h.to(ckpt_ptr.dtype.element_ty), mask=mask_h)

                p_ssm = (
                    ssm_state_ptr + state_idx * stride_ssm_slot + i_h * V * K
                    + (i_c * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
                )
                gl.store(p_ssm, b_h.to(ssm_state_ptr.dtype.element_ty), mask=mask_h)

    # ================================================================
    # IS_SPEC: pre-load conv history into registers
    # ================================================================
    if IS_SPEC:
        cs_off = i_t_start
        p_csq_base = p_cs + (q_ch_off + o_k) * stride_cs_dim
        p_csk_base = p_cs + (k_ch_off + o_k) * stride_cs_dim
        p_csv_base = p_cs + (v_ch_off + o_v_full) * stride_cs_dim
        b_colq0 = gl.load(p_csq_base + (cs_off + 0) * stride_cs_pos).to(tl.float32)
        b_colq1 = gl.load(p_csq_base + (cs_off + 1) * stride_cs_pos).to(tl.float32)
        b_colq2 = gl.load(p_csq_base + (cs_off + 2) * stride_cs_pos).to(tl.float32)
        b_colk0 = gl.load(p_csk_base + (cs_off + 0) * stride_cs_pos).to(tl.float32)
        b_colk1 = gl.load(p_csk_base + (cs_off + 1) * stride_cs_pos).to(tl.float32)
        b_colk2 = gl.load(p_csk_base + (cs_off + 2) * stride_cs_pos).to(tl.float32)
        b_colv0 = gl.load(p_csv_base + (cs_off + 0) * stride_cs_pos).to(tl.float32)
        b_colv1 = gl.load(p_csv_base + (cs_off + 1) * stride_cs_pos).to(tl.float32)
        b_colv2 = gl.load(p_csv_base + (cs_off + 2) * stride_cs_pos).to(tl.float32)

    # ================================================================
    # Hoist token-loop-invariant loads
    # ================================================================
    b_A = gl.load(A_log_ptr + i_h).to(tl.float32)
    b_dt = gl.load(dt_bias_ptr + i_h * K + o_k).to(tl.float32)

    if IS_SPEC:
        # Conv weights are constant across tokens — load once
        b_wq0 = gl.load(conv_weight_ptr + cw_q_base + o_k * stride_cw_ch + 0 * stride_cw_width).to(tl.float32)
        b_wq1 = gl.load(conv_weight_ptr + cw_q_base + o_k * stride_cw_ch + 1 * stride_cw_width).to(tl.float32)
        b_wq2 = gl.load(conv_weight_ptr + cw_q_base + o_k * stride_cw_ch + 2 * stride_cw_width).to(tl.float32)
        b_wq3 = gl.load(conv_weight_ptr + cw_q_base + o_k * stride_cw_ch + (W - 1) * stride_cw_width).to(tl.float32)
        b_wk0 = gl.load(conv_weight_ptr + cw_k_base + o_k * stride_cw_ch + 0 * stride_cw_width).to(tl.float32)
        b_wk1 = gl.load(conv_weight_ptr + cw_k_base + o_k * stride_cw_ch + 1 * stride_cw_width).to(tl.float32)
        b_wk2 = gl.load(conv_weight_ptr + cw_k_base + o_k * stride_cw_ch + 2 * stride_cw_width).to(tl.float32)
        b_wk3 = gl.load(conv_weight_ptr + cw_k_base + o_k * stride_cw_ch + (W - 1) * stride_cw_width).to(tl.float32)
        b_wv0 = gl.load(conv_weight_ptr + cw_v_base + o_v_full * stride_cw_ch + 0 * stride_cw_width).to(tl.float32)
        b_wv1 = gl.load(conv_weight_ptr + cw_v_base + o_v_full * stride_cw_ch + 1 * stride_cw_width).to(tl.float32)
        b_wv2 = gl.load(conv_weight_ptr + cw_v_base + o_v_full * stride_cw_ch + 2 * stride_cw_width).to(tl.float32)
        b_wv3 = gl.load(conv_weight_ptr + cw_v_base + o_v_full * stride_cw_ch + (W - 1) * stride_cw_width).to(tl.float32)

    # ================================================================
    # Per-token loop
    # ================================================================
    for i_t in range(seq_T):
        tok = bos + i_t
        p_x = x_ptr + tok * stride_x_tok

        # ============================================================
        # Phase 1: Conv1d Q + K + V
        # ============================================================
        if IS_SPEC:
            b_x_q = gl.load(p_x + q_ch_off + o_k).to(tl.float32)
            b_x_k = gl.load(p_x + k_ch_off + o_k).to(tl.float32)
            b_x_v = gl.load(p_x + v_ch_off + o_v_full).to(tl.float32)

            b_q = b_colq0 * b_wq0 + b_colq1 * b_wq1 + b_colq2 * b_wq2 + b_x_q * b_wq3
            b_q = b_q * tl.sigmoid(b_q)

            b_k = b_colk0 * b_wk0 + b_colk1 * b_wk1 + b_colk2 * b_wk2 + b_x_k * b_wk3
            b_k = b_k * tl.sigmoid(b_k)

            b_v_full = b_colv0 * b_wv0 + b_colv1 * b_wv1 + b_colv2 * b_wv2 + b_x_v * b_wv3
            b_v_full = b_v_full * tl.sigmoid(b_v_full)

            b_colq0 = b_colq1
            b_colq1 = b_colq2
            b_colq2 = b_x_q
            b_colk0 = b_colk1
            b_colk1 = b_colk2
            b_colk2 = b_x_k
            b_colv0 = b_colv1
            b_colv1 = b_colv2
            b_colv2 = b_x_v

            smem_v.store(b_v_full)

        else:
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

        b_a = gl.load(gate_ptr + (tok * H + i_h) * K + o_k).to(tl.float32)
        b_g = lower_bound * tl.sigmoid(tl.exp(b_A) * (b_a + b_dt))

        b_beta = tl.sigmoid(
            gl.load(beta_ptr + tok * stride_beta_tok + i_h).to(tl.float32)
        )

        # IS_SPEC per-token snapshot index
        if IS_SPEC and not USE_REPLAY:
            final_idx = gl.load(
                state_indices_ptr + i_n * stride_si_seq + i_t * stride_si_tok
            ).to(tl.int64)

        # ============================================================
        # Phase 2: Delta Rule (V in chunks, V read from LDS)
        # ============================================================
        # Hoist layout conversions + exp outside V-chunk loop (invariant across chunks)
        b_g_2d = gl.convert_layout(b_g, layout=gl.SliceLayout(0, hk_layout))
        b_exp_g = tl.exp(b_g_2d)
        b_k_2d = gl.convert_layout(b_k, layout=hk_k_slice)
        b_q_2d = gl.convert_layout(b_q, layout=hk_k_slice)

        o_sumsq = 0.0
        p_state_base = ssm_state_ptr + state_idx * stride_ssm_slot + i_h * V * K

        for i_c in gl.static_range(N_CHUNKS):
            o_v = i_c * CHUNK_V + gl.arange(0, CHUNK_V, layout=cv_layout)
            mask_v = o_v < V

            b_v = smem_v.slice(i_c * CHUNK_V, CHUNK_V).load(layout=cv_layout)

            o_v_2d = gl.arange(0, CHUNK_V, layout=gl.SliceLayout(1, hk_layout))
            o_k_2d = gl.arange(0, K, layout=gl.SliceLayout(0, hk_layout))
            mask_h = (o_v_2d[:, None] < V) & (o_k_2d[None, :] < K)
            p_h = p_state_base + (i_c * CHUNK_V + o_v_2d)[:, None] * K + o_k_2d[None, :]
            b_h = gl.load(p_h, mask=mask_h, other=0.0).to(tl.float32)

            b_h = b_h * b_exp_g[None, :]

            b_dot = gl.sum(b_h * b_k_2d[None, :], axis=1)
            b_dot_cv = gl.convert_layout(b_dot, layout=cv_layout)
            b_v = b_v - b_dot_cv
            b_v = b_v * b_beta
            b_v_2d = gl.convert_layout(b_v, layout=hk_v_slice)
            b_h = b_h + b_v_2d[:, None] * b_k_2d[None, :]

            # USE_REPLAY: write u record for this V-chunk
            if USE_REPLAY:
                pos = base + i_t
                gl.store(
                    buf_u_ptr
                    + slot * stride_bufu_slot
                    + i_h * stride_bufu_hv
                    + pos * stride_bufu_pos
                    + (i_c * CHUNK_V + o_v),
                    b_v.to(buf_u_ptr.dtype.element_ty),
                    mask=mask_v,
                )

            b_o = gl.sum(b_h * b_q_2d[None, :], axis=1)
            b_o_cv = gl.convert_layout(b_o, layout=cv_layout)

            # State write-back
            if USE_REPLAY:
                gl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)
            else:
                gl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)
                if IS_SPEC:
                    if final_idx >= 0:
                        p_snap = (
                            ssm_state_ptr
                            + final_idx * stride_ssm_slot
                            + i_h * V * K
                            + (i_c * CHUNK_V + o_v_2d)[:, None] * K
                            + o_k_2d[None, :]
                        )
                        gl.store(p_snap, b_h.to(ssm_state_ptr.dtype.element_ty), mask=mask_h)

            o_sumsq = o_sumsq + gl.sum(b_o_cv * b_o_cv)
            smem_out.slice(i_c * CHUNK_V, CHUNK_V).store(b_o_cv)

        # USE_REPLAY: write k and g records (once per token, after V-chunk loop)
        if USE_REPLAY:
            pos = base + i_t
            gl.store(
                buf_k_ptr
                + slot * stride_bufk_slot
                + i_h * stride_bufk_hv
                + pos * stride_bufk_pos
                + o_k,
                b_k.to(buf_k_ptr.dtype.element_ty),
            )
            gl.store(
                buf_g_ptr
                + slot * stride_bufg_slot
                + i_h * stride_bufg_hv
                + pos * stride_bufg_pos
                + o_k,
                b_g.to(buf_g_ptr.dtype.element_ty),
            )

        # ============================================================
        # Phase 3: Gated RMSNorm (read raw output from LDS)
        # ============================================================
        rstd = tl.math.rsqrt(o_sumsq / V + norm_eps)

        for i_c in gl.static_range(N_CHUNKS):
            o_v = i_c * CHUNK_V + gl.arange(0, CHUNK_V, layout=cv_layout)
            mask_v = o_v < V
            b_raw = smem_out.slice(i_c * CHUNK_V, CHUNK_V).load(layout=cv_layout)
            b_w = gl.load(norm_weight_ptr + o_v, mask=mask_v, other=0.0).to(tl.float32)
            b_og = gl.load(
                out_gate_ptr + tok * stride_og_tok + i_h * V + o_v,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            b_y = b_raw * rstd * b_w * tl.sigmoid(b_og)
            p_out_c = out_ptr + tok * (H * V) + i_h * V + o_v
            gl.store(p_out_c, b_y.to(out_ptr.dtype.element_ty), mask=mask_v)

    # ================================================================
    # IS_SPEC: bulk conv_state write-back after the token loop.
    # ================================================================
    if IS_SPEC:
        p_csq_base = p_cs + (q_ch_off + o_k) * stride_cs_dim
        p_csk_base = p_cs + (k_ch_off + o_k) * stride_cs_dim
        p_csv_base = p_cs + (v_ch_off + o_v_full) * stride_cs_dim
        val = STATE_LEN - seq_T
        for idx in gl.static_range(STATE_LEN):
            if idx + seq_T < STATE_LEN:
                src_pos = (i_t_start + 1 + idx) * stride_cs_pos
                gl.store(
                    p_csq_base + idx * stride_cs_pos,
                    gl.load(p_csq_base + src_pos),
                )
                gl.store(
                    p_csk_base + idx * stride_cs_pos,
                    gl.load(p_csk_base + src_pos),
                )
                gl.store(
                    p_csv_base + idx * stride_cs_pos,
                    gl.load(p_csv_base + src_pos),
                )
            else:
                x_tok = bos + (idx - val)
                p_x_tok = x_ptr + x_tok * stride_x_tok
                gl.store(
                    p_csq_base + idx * stride_cs_pos,
                    gl.load(p_x_tok + q_ch_off + o_k).to(p_csq_base.dtype.element_ty),
                )
                gl.store(
                    p_csk_base + idx * stride_cs_pos,
                    gl.load(p_x_tok + k_ch_off + o_k).to(p_csk_base.dtype.element_ty),
                )
                gl.store(
                    p_csv_base + idx * stride_cs_pos,
                    gl.load(p_x_tok + v_ch_off + o_v_full).to(p_csv_base.dtype.element_ty),
                )


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
    is_spec: bool = False,
    num_accepted_tokens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    state_len: int | None = None,
    conv_state_indices: torch.Tensor | None = None,
    use_replay: bool = False,
    ckpt: torch.Tensor | None = None,
    buf_k: torch.Tensor | None = None,
    buf_u: torch.Tensor | None = None,
    buf_g: torch.Tensor | None = None,
    write_pos: torch.Tensor | None = None,
    slot_idx: torch.Tensor | None = None,
    cap: int | None = None,
    bh: int | None = None,
) -> torch.Tensor:
    """Fused KDA decode: conv1d + recurrence + gated RMSNorm.

    Four modes:
      use_replay=False, is_spec=False → normal decode
      use_replay=False, is_spec=True  → DSpark spec decode (per-token snapshot)
      use_replay=True,  is_spec=False → ReplaySSM (ckpt rebuild + ring buffer)
      use_replay=True,  is_spec=True  → DSpark + ReplaySSM
    """
    T = mixed_qkv.shape[0]
    K = V = head_dim
    H = num_local_heads
    W = conv_weight.shape[-2] if conv_weight.dim() == 3 else conv_weight.shape[-1]
    lp = H * K
    CHUNK_V = min(triton.next_power_of_2(V), 32)
    N_CHUNKS = triton.cdiv(V, CHUNK_V)
    batch = cu_seqlens.shape[0] - 1
    out = torch.empty(T, lp, dtype=torch.bfloat16, device=mixed_qkv.device)

    STATE_LEN = state_len if state_len is not None else conv_state.shape[2]

    if conv_weight.dim() == 3:
        stride_cw_group = conv_weight.stride(0)
        stride_cw_width = conv_weight.stride(1)
        stride_cw_ch = conv_weight.stride(2)
    else:
        stride_cw_group = lp * conv_weight.stride(0)
        stride_cw_width = conv_weight.stride(1)
        stride_cw_ch = conv_weight.stride(0)

    if beta.dim() == 3:
        stride_beta_tok = beta.stride(1)
    else:
        stride_beta_tok = beta.stride(0)

    stride_og_tok = out_gate.stride(0)

    dev = mixed_qkv.device

    # --- Spec decode ---
    if not is_spec:
        num_accepted_tokens = torch.empty(0, dtype=torch.int64, device=dev)
        state_indices = torch.empty(0, dtype=torch.int32, device=dev)
        stride_si_seq = 0
        stride_si_tok = 0
    else:
        assert num_accepted_tokens is not None
        assert state_indices is not None
        stride_si_seq = state_indices.stride(0)
        stride_si_tok = state_indices.stride(1)

    # --- conv_state_indices ---
    if conv_state_indices is None:
        if is_spec or use_replay:
            conv_state_indices = ssm_state_indices.to(torch.int64)
        else:
            conv_state_indices = torch.empty(0, dtype=torch.int64, device=dev)

    # --- Replay SSM ---
    if not use_replay:
        ckpt = torch.empty(0, dtype=torch.float32, device=dev)
        buf_k = torch.empty(0, dtype=torch.float32, device=dev)
        buf_u = torch.empty(0, dtype=torch.float32, device=dev)
        buf_g = torch.empty(0, dtype=torch.float32, device=dev)
        write_pos = torch.empty(0, dtype=torch.int32, device=dev)
        slot_idx = torch.empty(0, dtype=torch.int32, device=dev)
        stride_ckpt_slot = 0
        stride_bufk_slot = stride_bufk_hv = stride_bufk_pos = 0
        stride_bufu_slot = stride_bufu_hv = stride_bufu_pos = 0
        stride_bufg_slot = stride_bufg_hv = stride_bufg_pos = 0
        CAP_val = 1
        BH_val = 1
    else:
        assert ckpt is not None
        assert buf_k is not None and buf_u is not None and buf_g is not None
        assert write_pos is not None and slot_idx is not None
        assert cap is not None and bh is not None
        stride_ckpt_slot = ckpt.stride(0)
        stride_bufk_slot = buf_k.stride(0)
        stride_bufk_hv = buf_k.stride(1)
        stride_bufk_pos = buf_k.stride(2)
        stride_bufu_slot = buf_u.stride(0)
        stride_bufu_hv = buf_u.stride(1)
        stride_bufu_pos = buf_u.stride(2)
        stride_bufg_slot = buf_g.stride(0)
        stride_bufg_hv = buf_g.stride(1)
        stride_bufg_pos = buf_g.stride(2)
        CAP_val = cap
        BH_val = bh

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
        num_accepted_tokens,
        state_indices,
        conv_state_indices,
        ckpt,
        buf_k,
        buf_u,
        buf_g,
        write_pos,
        slot_idx,
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
        STATE_LEN=STATE_LEN,
        CAP=CAP_val,
        BH=BH_val,
        USE_REPLAY=use_replay,
        IS_SPEC=is_spec,
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
        stride_si_seq=stride_si_seq,
        stride_si_tok=stride_si_tok,
        stride_ckpt_slot=stride_ckpt_slot,
        stride_bufk_slot=stride_bufk_slot,
        stride_bufk_hv=stride_bufk_hv,
        stride_bufk_pos=stride_bufk_pos,
        stride_bufu_slot=stride_bufu_slot,
        stride_bufu_hv=stride_bufu_hv,
        stride_bufu_pos=stride_bufu_pos,
        stride_bufg_slot=stride_bufg_slot,
        stride_bufg_hv=stride_bufg_hv,
        stride_bufg_pos=stride_bufg_pos,
        num_warps=4,
    )
    return out
