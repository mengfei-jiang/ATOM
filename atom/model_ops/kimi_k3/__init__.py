# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused model operations for Kimi-K3."""

from atom.model_ops.kimi_k3.activations import rmsnorm_gated, situ_and_mul
from atom.model_ops.kimi_k3.attention_residual import apply_attn_res
from atom.model_ops.kimi_k3.fused_kda_decode_gluon import fused_kda_decode_gluon
from atom.model_ops.kimi_k3.kda_state import gather_kda_initial_state

__all__ = [
    "apply_attn_res",
    "fused_kda_decode_gluon",
    "gather_kda_initial_state",
    "rmsnorm_gated",
    "situ_and_mul",
]
