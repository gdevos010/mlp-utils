
# mlp_utils

A collection of utilities for multi-layer perceptron models.

## Installation

```bash

pip install mlp-utils

```

## Features

### Activations

A collection of activation functions for MLPs.

- `ReluSquared`: `max(0, x)^2`, with an option to be signed.
- `gelu2`: `GELU(x)^2`
- `BSiLU`: `(x + α) * sigmoid(x) - α / 2`
- `NeLU`: `-α / (1 + x^2)`, often used as a backward function in STE.
- `Sugar`: A straight-through estimator that uses the backward function only for the negative part of the input.
- `StraightThroughEstimator`: A generic straight-through estimator that can be configured with different forward and backward passes.
- `ReluNelu`: An activation that uses ReLU in the forward pass and NeLU in the backward pass for the negative part, using the `Sugar` module.
- `sugar_relu`: A straight-through estimator with a ReLU forward pass and a sigmoid backward pass.

### Initialization

Utilities for initializing weights in neural network layers.

- `initialize_weights`: Initializes weights of a module with strategies like "gating" or "feedforward".
- `apply_initialization`: Applies initialization to all modules in a model.
- `create_initializer`: Creates a customized initializer function.

### Layers

#### FeedForward

A feed-forward block with optional GLU variants.

```python
from mlp_utils.layers.feedforward import FeedForward

ffn = FeedForward(
    dim=256,
    mult=4,
    glu_variant="swiglu",
)
```

#### nGPT

The `nGPTFeedForward` class implements the feed-forward block from the paper "nGPT: Normalized Transformer with Representation Learning on the Hypersphere."

```python
from mlp_utils.layers.ngpt import nGPTFeedForward


ngpt_ffn = nGPTFeedForward(
    dim=256,
    mult=4,
    glu_variant="swiglu",
)

```

#### Gating

A standardized gating mechanism.

```python
from mlp_utils.layers.gating import GatingMechanism

gate = GatingMechanism(
    input_dim=256,
    bottleneck_factor=0.5,
)
```

#### GLU Variants

Gated Linear Units (GLUs) are feed-forward layers that use multiplicative gating to improve model expressivity. Each GLU variant uses a different activation function for the gate.

- `GLU`: Gated Linear Unit with sigmoid activation
- `Bilinear`: GLU with no activation (identity)
- `ReGLU`: GLU with ReLU activation
- `SwiGLU`: GLU with Swish (SiLU) activation - commonly used in modern LLMs
- `GeGLU`: GLU with GELU activation

```python
from mlp_utils.layers.glu import SwiGLU

# Standard GLU with separate gate and value projections
swiglu = SwiGLU(
    dim_in=256,
    dim_out=512,
    bias=True,
)
```

#### Masked Gated Linear Units (MGLU)

MGLUs are a memory-efficient variant of GLUs that use a single shared weight matrix for both gate and value projections, with learnable binary masks to differentiate between them. This reduces memory bandwidth requirements during inference while maintaining the expressive power of traditional GLUs.

Based on the paper "Masked Gated Linear Unit" by Tajima et al. (2025), MGLUs implement a "Mixture of Element-wise Gating" (MoEG) architecture that can provide significant memory and computational benefits when implemented with optimized kernels like FlashMGLU.

Available MGLU variants:

- `MGLU`: Masked GLU with sigmoid activation
- `BilinearMGLU`: Masked GLU with no activation (identity)
- `ReMGLU`: Masked GLU with ReLU activation
- `SwiMGLU`: Masked GLU with Swish (SiLU) activation
- `GeMGLU`: Masked GLU with GELU activation

```python
from mlp_utils.layers.glu import SwiMGLU

# Memory-efficient MGLU with shared weight matrix
swimglu = SwiMGLU(
    dim_in=256,
    dim_out=512,
    bias=True,
)

# The mask parameter is learned during training
print(f"Learnable mask shape: {swimglu.mask.shape}")  # (512,)
```

**Note:** This implementation provides the MGLU architecture in PyTorch. The significant performance gains reported in the paper require optimized CUDA kernels like FlashMGLU.

#### MLP

A standardized MLP module.

```python

from mlp_utils.layers.mlp import MLP

mlp = MLP(
    input_dim=256,
    output_dim=256,
)

```

#### gMLP

The `GMLP` class implements the gMLP model from the paper "Pay Attention to MLPs."

```python
from mlp_utils.layers.gmlp import GMLP

gmlp = GMLP(
    dim=256,
    seq_len=64,
    depth=6,
)
```

#### Normalization

- `UnitNorm`: Normalizes a tensor to have a unit L2 norm along a given dimension.

#### Residual

- `ResidualWrapper`: Adds a residual connection to any module.

## Citations

```bibtex
@article{Zhang2024ReLU2WD,
    title   = {ReLU2 Wins: Discovering Efficient Activation Functions for Sparse LLMs},
    author  = {Zhengyan Zhang and Yixin Song and Guanghui Yu and Xu Han and Yankai Lin and Chaojun Xiao and Chenyang Song and Zhiyuan Liu and Zeyu Mi and Maosong Sun},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.03804},
    url     = {https://api.semanticscholar.org/CorpusID:267499856}
}
```

```bibtex
@inproceedings{Horuz2025TheRO,
    title   = {The Resurrection of the ReLU},
    author  = {Cocsku Can Horuz and Geoffrey Kasenbacher and Saya Higuchi and Sebastian Kairat and Jendrik Stoltz and Moritz Pesl and Bernhard A. Moser and Christoph Linse and Thomas Martinetz and Sebastian Otte},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:278959515}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}
}
```

```bibtex
@misc{tajima2025maskedgatedlinearunit,
      title ={Masked Gated Linear Unit},
      author={Yukito Tajima and Nakamasa Inoue and Yusuke Sekikawa and Ikuro Sato and Rio Yokota},
      year  ={2025},
      url   ={https://arxiv.org/abs/2506.23225},
}
```

```bibtex
@misc{loshchilov2025ngptnormalizedtransformerrepresentation,
      title={nGPT: Normalized Transformer with Representation Learning on the Hypersphere},
      author={Ilya Loshchilov and Cheng-Ping Hsieh and Simeng Sun and Boris Ginsburg},
      year={2025},
      url={https://arxiv.org/abs/2410.01131},
}
```

```bibtex
@misc{liu2021payattentionmlps,
      title={Pay Attention to MLPs},
      author={Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
      year={2021},
      url={https://arxiv.org/abs/2105.08050},
}
```

```bibtex
@misc{belcak2023fastfeedforwardnetworks,
      title={Fast Feedforward Networks}, 
      author={Peter Belcak and Roger Wattenhofer},
      year={2023},
      url={https://arxiv.org/abs/2308.14711}, 
}
```
