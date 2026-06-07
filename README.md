# mloptim

Optimizers and learning-rate schedulers for Rust ML — Adam, AdamW, SGD, gradient clipping, StepLR, CosineAnnealingLR, WarmupCosineScheduler.

All optimizers work directly with `mlautograd::Tensor` and read gradients from the thread-local gradient tape.

## Use cases

### LLM pre-training: AdamW with warmup
Pre-training transformers requires a brief linear warmup followed by cosine decay — the standard GPT/LLaMA recipe.

```rust
use mloptim::{AdamW, WarmupCosineScheduler, LRScheduler};

let mut opt = AdamW::new(3e-4).with_weight_decay(0.1);
let mut sched = WarmupCosineScheduler::new(3e-4, 2000, 100_000, 1e-5);

// each training step:
opt.step(&mut params)?;
sched.step(&mut opt);
```

### Fine-tuning: Adam with step decay
Supervised fine-tuning typically uses a flat LR with periodic step drops.

```rust
use mloptim::{Adam, StepLR, LRScheduler};

let mut opt = Adam::new(1e-4);
let mut sched = StepLR::new(1e-4, 10, 0.5); // halve every 10 epochs

for epoch in 0..50 {
    opt.step(&mut params)?;
    sched.step(&mut opt);
}
```

### Gradient stability: clip before step
Clip exploding gradients before the optimizer update to prevent NaN parameter values.

```rust
use mloptim::{AdamW, clip_grad_norm};

let norm = clip_grad_norm(&param_refs, 1.0);
log::debug!("grad norm: {norm}");
opt.step(&mut params)?;
```

### Simple training loops: SGD with momentum
Classic SGD with Nesterov-style momentum for small models or curriculum learning phases.

```rust
use mloptim::SGD;

let mut opt = SGD::new(0.01).with_momentum(0.9);
opt.step(&mut params)?;
```

## Crate layout

| Module | Contents |
|---|---|
| `optimizer` | `Optimizer` trait |
| `lr_scheduler` | `LRScheduler` trait |
| `optimizers::adam` | `Adam` (L2 weight decay, bias correction) |
| `optimizers::adamw` | `AdamW` (decoupled weight decay) |
| `optimizers::sgd` | `SGD` (with optional momentum) |
| `optimizers::grad_clip` | `clip_grad_norm`, `clip_grad_value` |
| `schedulers::step_lr` | `StepLR` |
| `schedulers::cosine_annealing_lr` | `CosineAnnealingLR` |
| `schedulers::warmup_cosine_scheduler` | `WarmupCosineScheduler` |
