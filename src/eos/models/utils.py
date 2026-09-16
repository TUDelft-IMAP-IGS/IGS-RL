"""Shared neural-network utilities for model construction.

Provides :func:`layer_init`, which applies orthogonal weight
initialisation — a common best practice for policy-gradient methods
that helps stabilise early training.
"""

import math

import torch
import torch.nn as nn


def layer_init(
    layer: nn.Module, std: float = math.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:  # Safety check
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
