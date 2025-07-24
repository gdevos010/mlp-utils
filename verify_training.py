import logging
import time

import torch
import torch.nn.functional as F

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from torch import nn

from mlp_utils.activations import BSiLU, Gelu2, ReluNelu, ReluSquared, SugarReLU
from mlp_utils.layers.fastfeedforward import FastFeedForward
from mlp_utils.layers.feedforward import FeedForward

# ... existing code ...

def get_model(model_name, dim, config):
    if model_name == "mlp":
        return MLP(
            dim=dim,
            act_fn=config["act_fn"],
            residual=config.get("residual", False),
            use_norm=config.get("use_norm", True),
            pre_norm=config.get("pre_norm", False),
        )
    if model_name == "feedforward":
        return FeedForward(
            dim=dim,
            mult=4,
            glu_variant=config["glu_variant"],
            activation=config.get("activation", nn.GELU),
        )
    if model_name == "fastfeedforward":
        return FastFeedForward(
            dim=dim,
            depth=3,
            mult=4,
            glu_variant=config["glu_variant"],
            expert_dim=config.get("expert_dim"),
        )
    if model_name == "ngpt":
        ff_net = FeedForward(dim=dim, mult=4, glu_variant="swiglu")
        return NGPT(
            dim=dim,
            depth=config["depth"],
            heads=config["heads"],
            ff_net=ff_net,
            scalar_alpha=config["scalar_alpha"],
        )

# ... existing code ...

configs = [
    {"model_name": "mlp", "act_fn": ReluSquared},
    {"model_name": "mlp", "act_fn": Gelu2},
    {"model_name": "mlp", "act_fn": BSiLU},
    {"model_name": "mlp", "act_fn": ReluNelu()},
    {"model_name": "mlp", "act_fn": SugarReLU()},
    # MLP parameter variants
    {"model_name": "mlp", "act_fn": nn.GELU, "residual": True},
    {"model_name": "mlp", "act_fn": nn.GELU, "use_norm": False},
    # FeedForward variants
    {"model_name": "feedforward", "glu_variant": "mreglu"},
    {"model_name": "feedforward", "glu_variant": "mbilinear"},
    # FastFeedForward variants
    {
        "model_name": "fastfeedforward",
        "glu_variant": "swiglu",
    },
    {
        "model_name": "fastfeedforward",
        "glu_variant": "geglu",
    },
    {
        "model_name": "fastfeedforward",
        "glu_variant": "reglu",
    },
    {
        "model_name": "fastfeedforward",
        "glu_variant": "bilinear",
    },
    {
        "model_name": "fastfeedforward",
        "glu_variant": "mswiglu",
    },
    # nGPT variants
    {"model_name": "ngpt", "scalar_alpha": True},
    {"model_name": "ngpt", "scalar_alpha": False},
]

# ... existing code ...
