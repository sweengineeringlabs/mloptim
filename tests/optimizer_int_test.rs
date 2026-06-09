// @allow: no_mocks_in_integration — concrete trait implementor needed to test Optimizer contract
use mloptim::Optimizer;
use mlautograd::{MlResult, Tensor};

struct MockOptimizer {
    learning_rate: f32,
}

impl Optimizer for MockOptimizer {
    fn step(&mut self, _params: &mut [&mut Tensor]) -> MlResult<()> {
        Ok(())
    }
    fn lr(&self) -> f32 {
        self.learning_rate
    }
    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

// @covers: Optimizer::lr
#[test]
fn test_optimizer_trait_fn_lr_returns_configured_value() {
    let opt = MockOptimizer { learning_rate: 0.01 };
    assert!((opt.lr() - 0.01).abs() < f32::EPSILON);
}

// @covers: Optimizer::set_lr
#[test]
fn test_optimizer_trait_fn_set_lr_updates_learning_rate() {
    let mut opt = MockOptimizer { learning_rate: 0.01 };
    opt.set_lr(0.001);
    assert!((opt.lr() - 0.001).abs() < f32::EPSILON);
}
