from torch.nn import GELU, ReLU, Sigmoid, SiLU

from .activations import (
    BSiLU,
    Gelu2,
    NeLU,
    ReluNelu,
    ReluSquared,
    StraightThroughEstimator,
    Sugar,
    SugarReLU,
)
from .layers.fastfeedforward import FastFeedForward
from .layers.feedforward import FeedForward
from .layers.gating import GatingMechanism
from .layers.glu import (
    GLU,
    MGLU,
    Bilinear,
    BilinearMGLU,
    GeGLU,
    GeMGLU,
    ReGLU,
    ReMGLU,
    SwiGLU,
    SwiMGLU,
)
from .layers.gmlp import GMLP
from .layers.mlp import MLP
from .layers.ngpt import L2Norm
from .layers.switch_ffn import SwitchFFN

__all__ = [
    "MLP",
    "FastFeedForward",
    "SwitchFFN",
    "GatingMechanism",
    "L2Norm",
    "GMLP",
    "FeedForward",
    "GLU",
    "Bilinear",
    "ReGLU",
    "SwiGLU",
    "GeGLU",
    "MGLU",
    "BilinearMGLU",
    "ReMGLU",
    "SwiMGLU",
    "GeMGLU",
    "ReluSquared",
    "Gelu2",
    "BSiLU",
    "NeLU",
    "Sugar",
    "StraightThroughEstimator",
    "ReluNelu",
    "SugarReLU",
    "GELU",
    "ReLU",
    "Sigmoid",
    "SiLU",
]
