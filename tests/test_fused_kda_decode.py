# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Gluon fused KDA decode kernel.

Compares the fused kernel (conv1d + recurrence + gated RMSNorm) output
against the reference 3-kernel implementation.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for Triton/Gluon kernels"
)


# ------------------------------------------------------------------ #
# Shared helpers
# ------------------------------------------------------------------ #

def _make_inputs(T, Hloc, D, device="cuda", dtype=torch.bfloat16):
    """Create random inputs matching KDA decode shapes."""
    W = 4
    lp = Hloc * D
    num_slots = T + 2
    batch = T

    inputs = dict(
        mixed_qkv=torch.randn(T, 3 * lp, dtype=dtype, device=device),
        conv_weight=torch.randn(3 * lp, W, dtype=dtype, device=device) * 0.1,
        conv_state=torch.randn(num_slots, 3 * lp, W - 1, dtype=dtype, device=device) * 0.1,
        gate=torch.randn(1, T, Hloc, D, dtype=dtype, device=device) * 0.5,
        beta=torch.randn(1, T, Hloc, dtype=dtype, device=device),
        out_gate=torch.randn(T, lp, dtype=dtype, device=device),
        A_log=torch.randn(Hloc, dtype=dtype, device=device) * 0.1,
        dt_bias=torch.randn(lp, dtype=dtype, device=device) * 0.1,
        ssm_state=torch.randn(num_slots, Hloc, D, D, dtype=dtype, device=device) * 0.01,
        norm_weight=torch.ones(D, dtype=dtype, device=device),
        ssm_state_indices=torch.arange(batch, dtype=torch.int32, device=device),
        cu_seqlens=torch.arange(batch + 1, dtype=torch.int64, device=device),
    )
    return inputs


def _reference_kda_decode(
    mixed_qkv, conv_state, conv_weight, gate, beta, out_gate,
    A_log, dt_bias, ssm_state, ssm_state_indices, cu_seqlens,
    norm_weight, norm_eps, head_dim, num_local_heads, lower_bound,
):
    """Reference implementation using the original 3 separate kernels."""
    from einops import rearrange
    from atom.model_ops.fla_ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from atom.model_ops.kimi_k3.activations import rmsnorm_gated
    from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update

    T = mixed_qkv.shape[0]
    lp = num_local_heads * head_dim

    q, k, v = causal_conv1d_update(
        mixed_qkv, conv_state, conv_weight, lp, lp, None, "silu",
        conv_state_indices=ssm_state_indices, validate_data=False,
    )
    q = rearrange(q, "t (h d) -> 1 t h d", d=head_dim)
    k = rearrange(k, "t (h d) -> 1 t h d", d=head_dim)
    v = rearrange(v, "t (h d) -> 1 t h d", d=head_dim)

    out = torch.empty(T, num_local_heads, head_dim, dtype=q.dtype, device=q.device)
    fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=gate, b=beta, dt_bias=dt_bias,
        q=q, k=k, v=v, o=out, initial_state=ssm_state,
        inplace_final_state=True, cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True, is_kda=True, lower_bound=lower_bound,
    )

    gate_3d = rearrange(out_gate[:T], "t (h d) -> t h d", d=head_dim)
    out = rmsnorm_gated(out, norm_weight, gate_3d, norm_eps)
    return out


# ------------------------------------------------------------------ #
# Gluon fused kernel tests (BF16 output)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("Hloc", [2, 4, 8])
@pytest.mark.parametrize("D", [128])
def test_gluon_fused_kda_decode_matches_reference(T, Hloc, D):
    from atom.model_ops.kimi_k3.fused_kda_decode_gluon import fused_kda_decode_gluon
    from einops import rearrange

    torch.manual_seed(42)
    inputs = _make_inputs(T, Hloc, D)
    norm_eps = 1e-6
    lower_bound = -5.0

    conv_state_ref = inputs["conv_state"].clone()
    ssm_state_ref = inputs["ssm_state"].clone()
    conv_state_fused = inputs["conv_state"].clone()
    ssm_state_fused = inputs["ssm_state"].clone()

    out_ref = _reference_kda_decode(
        inputs["mixed_qkv"], conv_state_ref, inputs["conv_weight"],
        inputs["gate"], inputs["beta"], inputs["out_gate"],
        inputs["A_log"], inputs["dt_bias"], ssm_state_ref,
        inputs["ssm_state_indices"], inputs["cu_seqlens"],
        inputs["norm_weight"], norm_eps, D, Hloc, lower_bound,
    )
    out_ref_flat = rearrange(out_ref, "t h d -> t (h d)")

    out_fused = fused_kda_decode_gluon(
        mixed_qkv=inputs["mixed_qkv"], conv_state=conv_state_fused,
        conv_weight=inputs["conv_weight"], gate=inputs["gate"],
        beta=inputs["beta"], out_gate=inputs["out_gate"],
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        ssm_state=ssm_state_fused,
        ssm_state_indices=inputs["ssm_state_indices"],
        cu_seqlens=inputs["cu_seqlens"],
        norm_weight=inputs["norm_weight"], norm_eps=norm_eps,
        head_dim=D, num_local_heads=Hloc, lower_bound=lower_bound,
    )

    torch.testing.assert_close(
        out_fused.float(), out_ref_flat.float(), atol=0.02, rtol=0.01,
        msg="Gluon fused output diverges from reference",
    )
    torch.testing.assert_close(
        ssm_state_fused, ssm_state_ref, atol=1e-3, rtol=1e-3,
        msg="SSM state diverges (Gluon)",
    )
    torch.testing.assert_close(
        conv_state_fused, conv_state_ref, atol=1e-3, rtol=1e-3,
        msg="Conv state diverges (Gluon)",
    )


def test_gluon_fused_kda_decode_pad_slot():
    from atom.model_ops.kimi_k3.fused_kda_decode_gluon import fused_kda_decode_gluon

    torch.manual_seed(42)
    inputs = _make_inputs(1, 2, 128)
    ssm_before = inputs["ssm_state"].clone()
    conv_before = inputs["conv_state"].clone()

    inputs["ssm_state_indices"] = torch.tensor([-1], dtype=torch.int32, device="cuda")

    fused_kda_decode_gluon(
        mixed_qkv=inputs["mixed_qkv"], conv_state=inputs["conv_state"],
        conv_weight=inputs["conv_weight"], gate=inputs["gate"],
        beta=inputs["beta"], out_gate=inputs["out_gate"],
        A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
        ssm_state=inputs["ssm_state"],
        ssm_state_indices=inputs["ssm_state_indices"],
        cu_seqlens=inputs["cu_seqlens"],
        norm_weight=inputs["norm_weight"], norm_eps=1e-6,
        head_dim=128, num_local_heads=2, lower_bound=-5.0,
    )

    assert torch.equal(inputs["ssm_state"], ssm_before), "SSM state modified for PAD_SLOT_ID"
    assert torch.equal(inputs["conv_state"], conv_before), "Conv state modified for PAD_SLOT_ID"


# ------------------------------------------------------------------ #
# Gluon performance sanity check
# ------------------------------------------------------------------ #

def test_gluon_faster_than_3kernel():
    """Sanity check: Gluon fused should be faster than 3 separate kernels."""
    import time
    from einops import rearrange
    from atom.model_ops.kimi_k3.fused_kda_decode_gluon import fused_kda_decode_gluon
    from atom.model_ops.fla_ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
    from atom.model_ops.kimi_k3.activations import rmsnorm_gated
    from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update

    torch.manual_seed(42)
    D = 128; Hloc = 8; T = 1; batch = 64
    lp = Hloc * D
    inputs = _make_inputs(T * batch, Hloc, D)
    inputs["ssm_state_indices"] = torch.arange(batch, dtype=torch.int32, device="cuda")
    inputs["cu_seqlens"] = torch.arange(batch + 1, dtype=torch.int64, device="cuda")

    def run_3k():
        cs = inputs["conv_state"].clone(); ss = inputs["ssm_state"].clone()
        q, k, v = causal_conv1d_update(
            inputs["mixed_qkv"], cs, inputs["conv_weight"], lp, lp, None, "silu",
            conv_state_indices=inputs["ssm_state_indices"], validate_data=False)
        out = torch.empty(T * batch, Hloc, D, dtype=torch.bfloat16, device="cuda")
        fused_sigmoid_gating_delta_rule_update(
            A_log=inputs["A_log"], a=inputs["gate"], b=inputs["beta"],
            dt_bias=inputs["dt_bias"],
            q=rearrange(q, "t (h d)->1 t h d", d=D),
            k=rearrange(k, "t (h d)->1 t h d", d=D),
            v=rearrange(v, "t (h d)->1 t h d", d=D),
            o=out, initial_state=ss, inplace_final_state=True,
            cu_seqlens=inputs["cu_seqlens"],
            ssm_state_indices=inputs["ssm_state_indices"],
            use_qk_l2norm_in_kernel=True, is_kda=True, lower_bound=-5.0)
        return rmsnorm_gated(out, inputs["norm_weight"],
                            rearrange(inputs["out_gate"], "t (h d)->t h d", d=D), 1e-6)

    def run_gluon():
        cs = inputs["conv_state"].clone(); ss = inputs["ssm_state"].clone()
        return fused_kda_decode_gluon(
            mixed_qkv=inputs["mixed_qkv"], conv_state=cs,
            conv_weight=inputs["conv_weight"], gate=inputs["gate"],
            beta=inputs["beta"], out_gate=inputs["out_gate"],
            A_log=inputs["A_log"], dt_bias=inputs["dt_bias"],
            ssm_state=ss, ssm_state_indices=inputs["ssm_state_indices"],
            cu_seqlens=inputs["cu_seqlens"], norm_weight=inputs["norm_weight"],
            norm_eps=1e-6, head_dim=D, num_local_heads=Hloc, lower_bound=-5.0)

    # Warmup
    for _ in range(5):
        run_3k(); run_gluon()
    torch.cuda.synchronize()

    N = 50
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): run_3k()
    torch.cuda.synchronize()
    t_3k = (time.perf_counter() - t0) / N

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): run_gluon()
    torch.cuda.synchronize()
    t_gl = (time.perf_counter() - t0) / N

    speedup = t_3k / t_gl
    print(f"\n3-kernel: {t_3k*1000:.3f}ms, Gluon: {t_gl*1000:.3f}ms, Speedup: {speedup:.2f}x")
    assert speedup > 1.5, f"Gluon should be at least 1.5x faster, got {speedup:.2f}x"
