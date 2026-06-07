# mloptim

> **TLDR:** Gradient-based optimizer library — Adam, AdamW, SGD, gradient clipping, and LR scheduling for Rust ML. See [Overview](docs/README.md) for details.

## Table of Contents
- [Quick Start](#quick-start)
- [API](#api)
- [Documentation](#documentation)

## Quick Start

```rust
use mloptim::{Adam, Optimizer, WarmupCosineScheduler, LRScheduler};

let mut opt = Adam::new(&mut model.parameters_mut(), 1e-3);
// training loop
opt.zero_grad();
// ... backward ...
opt.step(&mut model.parameters_mut());
```

## API

| Type | Description |
|------|-------------|
| `Optimizer` | Implement to define a gradient-based parameter update rule |
| `LRScheduler` | Implement to define a learning rate schedule |
| `Adam` | Adaptive moment estimation optimizer (default choice for most tasks) |
| `AdamW` | Adam with decoupled weight decay (preferred for transformers) |
| `SGD` | Stochastic gradient descent with optional momentum |
| `clip_grad_norm` | Clip gradient L2 norm to a maximum value |
| `WarmupCosineScheduler` | Linear warmup then cosine annealing, used in LLM training |

## Documentation

- [Architecture](docs/3-design/architecture.md) - System design

## Related FRs

- None (foundational crate)
