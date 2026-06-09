# mloptim

> **TLDR:** Gradient-based optimizer library — Adam, AdamW, SGD, gradient clipping, and LR scheduling for Rust ML. See [Overview](docs/README.md) for details.

## Table of Contents
- [Quick Start](#quick-start)
- [API](#api)
- [Documentation](#documentation)
- [Development](#development)

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
| `GradClipper` | Clip gradients by L2 norm (`clip_grad_norm`) or element-wise value (`clip_grad_value`) |
| `WarmupCosineScheduler` | Linear warmup then cosine annealing, used in LLM training |

## Documentation

- [Overview](docs/README.md) - W³H
- [Architecture](docs/3-design/architecture.md) - System design
- [Integration](docs/3-design/integration.md) - Integration guide

## Development

After cloning, activate the commit-msg hook:

```sh
git config core.hooksPath .githooks
```

The hook rejects commits that carry AI-assistant attribution (Co-Authored-By trailers, "Generated with …" footers, 🤖 emoji). The same check runs server-side via `.github/workflows/no-ai-attribution.yml` and cannot be bypassed with `--no-verify`.

## Related FRs

- None (foundational crate)
