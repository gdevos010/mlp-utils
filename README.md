
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
- `Gelu2`: `GELU(x)^2`
- `BSiLU`: `(x + α) * sigmoid(x) - α / 2`
- `NeLU`: `-α / (1 + x^2)`, often used as a backward function in STE.
- `Sugar`: A straight-through estimator that uses the backward function only for the negative part of the input.
- `StraightThroughEstimator`: A generic straight-through estimator that can be configured with different forward and backward passes.
- `ReluNelu`: An activation that uses ReLU in the forward pass and NeLU in the backward pass for the negative part, using the `Sugar` module.
- `SugarReLU`: A straight-through estimator with a ReLU forward pass and a sigmoid backward pass.

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

#### NGPT

The `NGPT` class implements the feed-forward block from the paper ["nGPT: Normalized Transformer with Representation Learning on the Hypersphere."](https://arxiv.org/html/2410.01131v2).

This module applies the nGPT update rule, which involves normalizing hidden states and using a learnable interpolation parameter (`alpha_m`) to update the representation on the hypersphere. By default, it uses a `SwiGLU` feed-forward network with weight-normalized linear layers (`NormLinear`), making it a self-contained implementation of the nGPT MLP block.

You can use it as a standalone layer:

```python
from mlp_utils.layers import NGPT

# Initialize the nGPT feed-forward block
ngpt_feedforward = NGPT(
    dim=256,
)

# The resulting module can be used as a drop-in replacement for a standard feedforward
```

Alternatively, you can provide your own feed-forward network, which will be wrapped with the nGPT update rule:

```python
from mlp_utils.layers import FeedForward, NGPT

# 1. Create a custom feed-forward network
feedforward_net = FeedForward(
    dim=256,
    mult=4,
    glu_variant="swiglu",
)

# 2. Wrap it with the NGPT layer
ngpt_feedforward_wrapped = NGPT(
    feedforward_net=feedforward_net,
    dim=256,
)
```

#### FastFeedForward

The `FastFeedForward` class implements the Fast Feedforward Network from the paper "Fast Feedforward Networks" by Belcak and Wattenhofer. This layer uses a tree of routers to dynamically select a small subset of "expert" `FeedForward` networks for each input token, enabling conditional computation.

This architecture achieves logarithmic time complexity with respect to the number of experts, leading to significant efficiency gains while preserving a high degree of predictive performance.

```python
from mlp_utils.layers import FastFeedForward

# Create a FastFeedForward layer with a tree of depth 3 (2^3 = 8 experts)
fff = FastFeedForward(
    dim=256,
    depth=3,
    mult=4,
    glu_variant="swiglu",
)
```

#### PathWeightedFFF

The `PathWeightedFFF` class implements a hierarchical, path-dependent neural network that uses a binary tree structure. Unlike a Mixture-of-Experts (MoE) model that routes an input to a single expert, this network computes its output by combining transformations from *every* node along the traversed path.

The routing logits themselves are activated with GELU and used as weights to combine the transformations, allowing the model to learn hierarchical features in a path-dependent manner. This offers a different architectural trade-off compared to sparse MoE layers like `FastFeedForward`.

```python
from mlp_utils.layers import PathWeightedFFF

# Create a PathWeightedFFF layer with a tree of depth 4
pfff = PathWeightedFFF(
    input_width=256,
    depth=4,
    output_width=256,
)
```

#### SwitchFFN

The `SwitchFFN` layer implements the Switch Transformer feed-forward layer from the paper ["Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"](https://arxiv.org/abs/2101.03961).

This layer uses a Mixture-of-Experts (MoE) approach, where a router network dynamically sends each input token to one of several "expert" `FeedForward` networks. This allows for a massive number of parameters while keeping the computational cost per token constant. The layer also includes an auxiliary load-balancing loss to encourage experts to be utilized evenly.

```python
from mlp_utils.layers import SwitchFFN

# Create a SwitchFFN layer with 8 experts
switch_ffn = SwitchFFN(
    dim=256,
    num_experts=8,
    capacity_factor=1.25,
    ff_kwargs=dict(mult=4, glu_variant="swiglu"),
)

# The forward pass returns the output and the load-balancing loss
output, loss = switch_ffn(input_tensor)
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

- `L2Norm`: Normalizes a tensor to have a unit L2 norm along a given dimension.

#### Residual

- `ResidualWrapper`: Adds a residual connection to any module.

### a not very accurate benchmark using toy dataset
                                               
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Model           ┃ Compile ┃  Params ┃ Runtime (s) ┃ Configuration                                      ┃ Final Loss ┃ Status  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━┩
│ mlp             │ True    │  33.41K │       10.15 │ act_fn=GELU                                        │   0.221746 │ Success │
│ mlp             │ True    │  33.41K │        2.40 │ act_fn=ReLU                                        │   0.225678 │ Success │
│ mlp             │ True    │  33.41K │        2.39 │ act_fn=SiLU                                        │   0.215601 │ Success │
│ mlp             │ True    │  33.41K │        2.35 │ act_fn=ReluSquared                                 │   0.246249 │ Success │
│ mlp             │ True    │  33.41K │        2.53 │ act_fn=Gelu2                                       │   0.252107 │ Success │
│ mlp             │ True    │  33.41K │        2.44 │ act_fn=BSiLU                                       │   0.211360 │ Success │
│ mlp             │ True    │  33.41K │        2.51 │ act_fn=ReluNelu(                                   │   0.223811 │ Success │
│                 │         │         │             │   (forward_fn): ReLU()                             │            │         │
│                 │         │         │             │   (backward_fn): NeLU()                            │            │         │
│                 │         │         │             │ )                                                  │            │         │
│ mlp             │ True    │  33.41K │        2.38 │ act_fn=GELU, residual=True                         │   0.255551 │ Success │
│ mlp             │ True    │  33.09K │        1.82 │ act_fn=GELU, use_norm=False                        │   0.220111 │ Success │
│ mlp             │ True    │  33.22K │        2.38 │ act_fn=GELU, pre_norm=True                         │   0.221599 │ Success │
│ feedforward     │ True    │  33.09K │        1.76 │ glu_variant=none, activation=GELU                  │   0.207531 │ Success │
│ feedforward     │ True    │  49.73K │        1.90 │ glu_variant=glu                                    │   0.179443 │ Success │
│ feedforward     │ True    │  49.73K │        1.87 │ glu_variant=swiglu                                 │   0.306741 │ Success │
│ feedforward     │ True    │  49.73K │        1.89 │ glu_variant=geglu                                  │   0.304403 │ Success │
│ feedforward     │ True    │  49.73K │        1.82 │ glu_variant=reglu                                  │   0.293338 │ Success │
│ feedforward     │ True    │  49.73K │        1.70 │ glu_variant=bilinear                               │   0.318753 │ Success │
│ feedforward     │ True    │  33.34K │        1.76 │ glu_variant=mglu                                   │   0.189905 │ Success │
│ feedforward     │ True    │  33.34K │        1.79 │ glu_variant=mswiglu                                │   0.204473 │ Success │
│ feedforward     │ True    │  33.34K │        1.82 │ glu_variant=mgeglu                                 │   0.211235 │ Success │
│ feedforward     │ True    │  33.34K │        1.78 │ glu_variant=mreglu                                 │   0.213310 │ Success │
│ feedforward     │ True    │  33.34K │        1.73 │ glu_variant=mbilinear                              │   0.236762 │ Success │
│ fastfeedforward │ True    │ 398.28K │       18.49 │ glu_variant=swiglu, expert_dim=8                   │   0.176578 │ Success │
│ fastfeedforward │ True    │ 398.28K │       18.79 │ glu_variant=geglu, expert_dim=8                    │   0.176054 │ Success │
│ fastfeedforward │ True    │ 267.21K │       18.12 │ glu_variant=mswiglu, expert_dim=8                  │   0.173000 │ Success │
│ fastfeedforward │ True    │ 398.28K │       17.40 │ glu_variant=swiglu, expert_dim=8                   │   0.171305 │ Success │
│ fastfeedforward │ True    │ 398.28K │       17.44 │ glu_variant=swiglu, expert_dim=8                   │   0.174691 │ Success │
│ fastfeedforward │ True    │ 398.28K │       18.10 │ glu_variant=swiglu, expert_dim=8                   │   0.172715 │ Success │
│ pathweightedfff │ True    │  63.38K │       20.21 │ depth=3                                            │   0.246856 │ Success │
│ pathweightedfff │ True    │  63.38K │       20.27 │ depth=3, activation=silu                           │   0.250302 │ Success │
│ pathweightedfff │ True    │ 266.18K │       25.57 │ depth=5                                            │   0.237995 │ Success │
│ ngpt            │ True    │  49.73K │        2.48 │ scalar_alpha=True                                  │   0.005054 │ Success │
│ ngpt            │ True    │  49.79K │        2.40 │ scalar_alpha=False                                 │   0.005193 │ Success │
│ gmlp            │ True    │ 218.11K │       10.33 │                                                    │   0.049925 │ Success │
│ switch_ffn      │ True    │ 398.34K │       10.07 │ num_experts=8, ff_kwargs={'mult': 4,               │   0.274261 │ Success │
│                 │         │         │             │ 'glu_variant': 'swiglu'}                           │            │         │
│ switch_ffn      │ True    │ 399.36K │       17.40 │ num_experts=16, ff_kwargs={'mult': 2,              │   0.362581 │ Success │
│                 │         │         │             │ 'glu_variant': 'geglu'}                            │            │         │
└─────────────────┴─────────┴─────────┴─────────────┴────────────────────────────────────────────────────┴────────────┴─────────┘

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

```bibtex
@misc{fedus2022switchtransformersscalingtrillion,
      title={Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity}, 
      author={William Fedus and Barret Zoph and Noam Shazeer},
      year={2022},
      eprint={2101.03961},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2101.03961}, 
}
```

```bibtex
@misc{charalampopoulos2024enhancingfastfeedforward,
      title={Enhancing Fast Feed Forward Networks with Load Balancing and a Master Leaf Node}, 
      author={Andreas Charalampopoulos and Nikolas Chatzis and Foivos Ntoulas-Panagiotopoulos and Charilaos Papaioannou and Alexandros Potamianos},
      year={2024},
      eprint={2405.16836},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.16836}, 
}
```

```bibtex
@misc{belcak2023exponentiallyfasterlanguagemodelling,
      title={Exponentially Faster Language Modelling}, 
      author={Peter Belcak and Roger Wattenhofer},
      year={2023},
      url={https://arxiv.org/abs/2311.10770}, 
}
```
