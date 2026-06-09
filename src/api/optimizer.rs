use mlautograd::{MlResult, Tensor};

pub trait Optimizer {
    fn step(&mut self, params: &mut [&mut Tensor]) -> MlResult<()>;
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}
