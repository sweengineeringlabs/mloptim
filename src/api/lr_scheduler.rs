use crate::api::optimizer::Optimizer;

pub trait LRScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer);
    fn get_lr(&self) -> f32;
}
