use mloptim::{Adam, Optimizer};
use mlautograd::Tensor;

fn main() {
    let mut w = Tensor::randn([4, 4]);
    w.set_requires_grad(true);
    let mut opt = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let mut params = [&mut w];
    opt.step(&mut params).expect("optimizer step");
    println!("step completed, lr={}", opt.lr());
}
