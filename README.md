
# mlp_utils

A collection of utilities for multi-layer perceptron models.

## Installation

```bash
pip install -e ./mlp-utils
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

- `initialize_weights`: Initializes weights of a module with strategies like "default", "gating", "embedding", or "expert".
- `apply_initialization`: Applies initialization to all modules in a model.
- `create_initializer`: Creates a customized initializer function.

```python
from mlp_utils.layers.init_weights import (
    initialize_weights,
    apply_initialization,
    create_initializer,
)

# Create and apply a reusable initializer
initializer = create_initializer(init_method="default", nonlinearity="relu", scale=1.0)
model.apply(initializer)
```

### Layers

#### FeedForward

A feed-forward block with optional GLU variants.

```python
from mlp_utils.layers.feedforward import FeedForward

ffn = FeedForward(
    dim=256,
    mult=2,
    glu_variant="swiglu",
)
```

#### NGPT

The `NGPT` class implements the feed-forward block from the paper ["nGPT: Normalized Transformer with Representation Learning on the Hypersphere."](https://arxiv.org/html/2410.01131v2).

This module applies the nGPT update rule, which involves normalizing hidden states and using a learnable interpolation parameter (`alpha_m`) to update the representation on the hypersphere.

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
    dim_ff=1024,
    seq_len=64,
    depth=6,
)
```

#### Normalization

- `L2Norm`: Normalizes a tensor to have a unit L2 norm along a given dimension.

```python
from mlp_utils import L2Norm

norm = L2Norm(dim=-1)
```

#### Residual

- `ResidualWrapper`: Adds a residual connection to any module.

```python
from mlp_utils.layers import ResidualWrapper
import torch.nn as nn

residual_mlp = ResidualWrapper(
    nn.Sequential(
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 256),
    )
)
```

#### FiLM (Conditioning)

Feature-wise Linear Modulation (FiLM) conditions activations with per-feature scale and shift predicted from a conditioning signal.

```python
from mlp_utils.layers import FiLM, FiLMGenerator
import torch

# Shapes: x [B, T, D], cond [B, C] (global) or [B, T, C] (token-wise)
x = torch.randn(2, 8, 256)
cond = torch.zeros(2, 16)  # zero makes FiLM an identity at init

gen = FiLMGenerator(cond_dim=16, feature_dim=256, token_wise=False)
film = FiLM(feature_dim=256)

gamma, beta = gen(cond)          # [B, D]
y = film(x, gamma, beta)         # [B, T, D]
```

##### ResidualFiLM (pre-norm site)

Wrap any module with a pre-norm residual FiLM hook. Zero conditioning is an exact no-op.

```python
from mlp_utils.layers import ResidualFiLM, FiLMGenerator
import torch.nn as nn
import torch

dim = 256
module = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
gen = FiLMGenerator(cond_dim=32, feature_dim=dim, token_wise=False)
wrapped = ResidualFiLM(module, feature_dim=dim, generator=gen)

x = torch.randn(2, dim)
cond = torch.zeros(2, 32)
out = wrapped(x, cond)
```

##### FFNFiLM (gate the FFN hidden state)

Apply FiLM to an FFN's intermediate activation for strong, stable control.

```python
from mlp_utils.layers import FFNFiLM, FiLMGenerator
import torch

dim = 256
gen = FiLMGenerator(cond_dim=32, feature_dim=dim * 4, token_wise=False)
ffn_film = FFNFiLM(dim=dim, hidden_mult=4, generator=gen)

x = torch.randn(2, dim)
cond = torch.randn(2, 32)
out = ffn_film(x, cond)
```

##### LowRankFiLM and per-layer generators

Use a small rank-K basis for FiLM to keep parameters tiny. Build shared or per-layer generators.

```python
from mlp_utils.layers import LowRankFiLM, FiLMGenerator, build_film_generators
import torch

rank = 4
dim = 256
lr_film = LowRankFiLM(feature_dim=dim, rank=rank)

# coeffs shape: [..., 2 * rank]; here token-wise over a sequence
coeffs = torch.zeros(2, 8, 2 * rank)
y = lr_film(torch.randn(2, 8, dim), coeffs)

# Build shared or per-layer FiLM generators
shared_gen = build_film_generators(
    shared=True,
    num_layers=6,
    factory=FiLMGenerator,
    cond_dim=32,
    feature_dim=dim,
    token_wise=False,
)

per_layer_gens = build_film_generators(
    shared=False,
    num_layers=6,
    factory=FiLMGenerator,
    cond_dim=32,
    feature_dim=dim,
    token_wise=False,
)
```

### a not very accurate benchmark using toy dataset
# MNIST Training Summary

| Model                        | Dim | Params  | Runtime (s) | Test Acc | Test Loss | Configuration                                                                 | Status  |
|------------------------------|-----|---------|-------------|----------|-----------|-------------------------------------------------------------------------------|---------|
| mlp                          | 158 | 205.41K | 92.91       | 78.67%   | 0.640622  | act_fn=GELU, budget=200.00K±5%, actual_params=205.41K                         | Success |
| mlp                          | 158 | 205.41K | 150.48      | 78.46%   | 0.656069  | act_fn=ReLU, budget=200.00K±5%, actual_params=205.41K                         | Success |
| mlp                          | 158 | 205.41K | 114.84      | 76.38%   | 0.725208  | act_fn=SiLU, budget=200.00K±5%, actual_params=205.41K                         | Success |
| mlp                          | 158 | 205.41K | 39.40       | 78.70%   | 0.651187  | act_fn=ReluSquared, budget=200.00K±5%, actual_params=205.41K                  | Success |
| mlp                          | 158 | 205.41K | 123.16      | 78.50%   | 0.645665  | act_fn=Gelu2, budget=200.00K±5%, actual_params=205.41K                         | Success |
| mlp                          | 158 | 205.41K | 86.55       | 69.66%   | 0.904601  | act_fn=BSiLU, budget=200.00K±5%, actual_params=205.41K                         | Success |
| mlp                          | 158 | 205.41K | 434.94      | 79.48%   | 0.631724  | act_fn=ReluNelu (forward_fn=ReLU, backward_fn=NeLU), budget=200.00K±5%        | Success |
| mlp                          | 158 | 205.41K | 259.04      | 78.62%   | 0.650378  | act_fn=GELU, residual=True, budget=200.00K±5%, actual_params=205.41K           | Success |
| mlp                          | 159 | 207.35K | 318.10      | 79.57%   | 0.600632  | act_fn=GELU, use_norm=False, budget=200.00K±5%, actual_params=207.35K          | Success |
| mlp                          | 159 | 207.50K | 117.76      | 80.37%   | 0.590114  | act_fn=GELU, pre_norm=True, budget=200.00K±5%, actual_params=207.50K           | Success |
| gmlp                         |  62 | 199.47K | 98.51       | 98.04%   | 0.062043  | budget=200.00K±5%, actual_params=199.47K                                      | Success |
| gmlp                         |  62 | 199.47K | 106.58      | 98.24%   | 0.056188  | canonical_gate=True, budget=200.00K±5%, actual_params=199.47K                  | Success |
| gmlp                         |  62 | 199.47K | 99.83       | 98.14%   | 0.056123  | drop_path=0.1, budget=200.00K±5%, actual_params=199.47K                        | Success |
| gmlp                         |  62 | 199.47K | 90.14       | 98.08%   | 0.062160  | gate_activation=SiLU, budget=200.00K±5%, actual_params=199.47K                 | Success |
| gmlp                         |  62 | 199.47K | 205.67      | 98.21%   | 0.063012  | canonical_gate=True, gate_activation=SiLU, budget=200.00K±5%, actual=199.47K   | Success |
| gmlp                         |  62 | 199.47K | 88.33       | 98.59%   | 0.044483  | canonical_gate=True, drop_path=0.1, budget=200.00K±5%, actual=199.47K          | Success |
| gmlp                         |  62 | 199.47K | 40.88       | 98.25%   | 0.056019  | canonical_gate=False, drop_path=0.1, budget=200.00K±5%, actual=199.47K         | Success |
| feedforward                  | 159 | 207.35K | 29.76       | 80.06%   | 0.610332  | glu_variant=none, activation=GELU, budget=200.00K±5%, actual=207.35K           | Success |
| feedforward                  | 126 | 195.06K | 29.32       | 81.01%   | 0.569265  | glu_variant=glu, budget=200.00K±5%, actual=195.06K                             | Success |
| feedforward                  | 126 | 195.06K | 33.05       | 82.86%   | 0.526871  | glu_variant=swiglu, budget=200.00K±5%, actual=195.06K                          | Success |
| feedforward                  | 126 | 195.06K | 13.49       | 79.33%   | 0.626115  | glu_variant=geglu, budget=200.00K±5%, actual=195.06K                           | Success |
| feedforward                  | 126 | 195.06K | 33.14       | 82.04%   | 0.545428  | glu_variant=reglu, budget=200.00K±5%, actual=195.06K                           | Success |
| feedforward                  | 126 | 195.06K | 11.28       | 61.64%   | 1.087680  | glu_variant=bilinear, budget=200.00K±5%, actual=195.06K                        | Success |
| residual_film_ffn            | 120 | 206.17K | 31.50       | 84.54%   | 0.482169  | budget=200.00K±5%, actual_params=206.17K                                      | Success |
| ffn_film                     | 108 | 190.09K | 22.24       | 77.93%   | 0.635031  | budget=200.00K±5%, actual_params=190.09K                                      | Success |
| ffn_lowrank_film             | 156 | 205.93K | 29.48       | 81.02%   | 0.571604  | budget=200.00K±5%, actual_params=205.93K                                      | Success |
| residual_film_stack_shared   |  70 | 190.41K | 33.20       | 86.41%   | 0.433458  | film_depth=3, budget=200.00K±5%, actual_params=190.41K                         | Success |
| residual_film_stack_perlayer |  67 | 192.57K | 19.25       | 85.47%   | 0.452107  | film_depth=3, budget=200.00K±5%, actual_params=192.57K                         | Success |
| feedforward                  | 158 | 205.41K | 37.62       | 79.34%   | 0.618405  | glu_variant=mglu, budget=200.00K±5%, actual=205.41K                            | Success |
| feedforward                  | 158 | 205.41K | 20.14       | 77.85%   | 0.672264  | glu_variant=mswiglu, budget=200.00K±5%, actual=205.41K                         | Success |
| feedforward                  | 158 | 205.41K | 20.06       | 77.80%   | 0.670321  | glu_variant=mgeglu, budget=200.00K±5%, actual=205.41K                          | Success |
| feedforward                  | 158 | 205.41K | 15.57       | 76.75%   | 0.699149  | glu_variant=mreglu, budget=200.00K±5%, actual=205.41K                          | Success |
| feedforward                  | 158 | 205.41K | 17.83       | 61.13%   | 1.090482  | glu_variant=mbilinear, budget=200.00K±5%, actual=205.41K                       | Success |
| fastfeedforward              |  46 | 208.03K | 9.48        | 41.38%   | 1.982514  | glu_variant=swiglu, budget=200.00K±5%, actual=208.03K                          | Success |
| fastfeedforward              |  46 | 208.03K | 15.46       | 63.97%   | 1.150880  | glu_variant=geglu, budget=200.00K±5%, actual=208.03K                           | Success |
| fastfeedforward              |  55 | 199.45K | 32.14       | 60.84%   | 1.314935  | glu_variant=mswiglu, budget=200.00K±5%, actual=199.45K                         | Success |
| pathweightedfff              | 111 | 191.17K | 58.14       | 68.43%   | 0.930329  | depth=3, budget=200.00K±5%, actual=191.17K                                     | Success |
| pathweightedfff              | 111 | 191.17K | 77.54       | 70.16%   | 0.879889  | depth=3, activation=silu, budget=200.00K±5%, actual=191.17K                    | Success |
| pathweightedfff              |  55 | 199.06K | 96.13       | 78.40%   | 0.659356  | depth=5, budget=200.00K±5%, actual=199.06K                                     | Success |
| ngpt                         | 126 | 195.06K | 112.29      | 80.55%   | 0.621386  | scalar_alpha=True, budget=200.00K±5%, actual=195.06K                           | Success |
| ngpt                         | 126 | 195.18K | 90.35       | 79.06%   | 0.678157  | scalar_alpha=False, budget=200.00K±5%, actual=195.18K                          | Success |
| switch_ffn                   |  63 | 195.25K | 22.12       | 70.33%   | 0.876992  | num_experts=8, ff_kwargs={mult:2, glu_variant:swiglu}, budget=200.00K±5%       | Success |
| switch_ffn                   |  46 | 208.80K | 124.47      | 74.18%   | 0.775046  | num_experts=16, ff_kwargs={mult:2, glu_variant:geglu}, budget=200.00K±5%       | Success |

## TODO

- [] [Enhancing Fast Feed Forward Networks with Load Balancing and a Master Leaf Node](https://arxiv.org/html/2405.16836v1)
- [] limiting soft routing to top-k paths (stochastic top-k or beam) to reduce compute work during training


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
<!--
```bibtex
# TODO
@misc{charalampopoulos2024enhancingfastfeedforward,
      title={Enhancing Fast Feed Forward Networks with Load Balancing and a Master Leaf Node},
      author={Andreas Charalampopoulos and Nikolas Chatzis and Foivos Ntoulas-Panagiotopoulos and Charilaos Papaioannou and Alexandros Potamianos},
      year={2024},
      eprint={2405.16836},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.16836},
}
``` -->

```bibtex
@misc{belcak2023exponentiallyfasterlanguagemodelling,
      title={Exponentially Faster Language Modelling},
      author={Peter Belcak and Roger Wattenhofer},
      year={2023},
      url={https://arxiv.org/abs/2311.10770},
}
```

```bibtex
@misc{perez2017filmvisualreasoninggeneral,
      title={FiLM: Visual Reasoning with a General Conditioning Layer},
      author={Ethan Perez and Florian Strub and Harm de Vries and Vincent Dumoulin and Aaron Courville},
      year={2017},
      url={https://arxiv.org/abs/1709.07871},
}
```
