use mlautograd::{MlResult, Tensor, TensorId, tape};
use crate::api::optimizer::Optimizer;
use std::collections::HashMap;

/// AdamW optimizer (Loshchilov & Hutter, 2017).
///
/// Implements Adam with *decoupled* weight decay. Weight decay is applied
/// directly to the parameters before the adaptive update step.
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    /// Global step counter (incremented once per `step()` call).
    t: u64,
    /// First moment estimates (mean of gradients).
    m: HashMap<TensorId, Tensor>,
    /// Second moment estimates (mean of squared gradients).
    v: HashMap<TensorId, Tensor>,
}

impl AdamW {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [&mut Tensor]) -> MlResult<()> {
        self.t += 1;
        let t = self.t;

        for param in params.iter_mut() {
            if let Some(grad) = tape::grad(param) {
                let param_id = param.id();
                let param_shape = param.shape().to_vec();

                // Decoupled weight decay: param -= lr * weight_decay * param
                if self.weight_decay > 0.0 {
                    let decay = param.mul_scalar_raw(self.learning_rate * self.weight_decay);
                    let decayed_param = param.sub_raw(&decay)?;
                    param.update_data_from(&decayed_param);
                }

                if !self.m.contains_key(&param_id) {
                    self.m.insert(param_id, Tensor::zeros(param_shape.clone()));
                }
                if !self.v.contains_key(&param_id) {
                    self.v.insert(param_id, Tensor::zeros(param_shape));
                }

                let m_prev = self.m.get(&param_id).unwrap();
                let v_prev = self.v.get(&param_id).unwrap();

                let m_new = m_prev
                    .mul_scalar_raw(self.beta1)
                    .add_raw(&grad.mul_scalar_raw(1.0 - self.beta1))?;

                let grad_sq = grad.mul_raw(&grad)?;
                let v_new = v_prev
                    .mul_scalar_raw(self.beta2)
                    .add_raw(&grad_sq.mul_scalar_raw(1.0 - self.beta2))?;

                let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
                let m_hat = m_new.div_scalar_raw(bias_correction1);

                let bias_correction2 = 1.0 - self.beta2.powi(t as i32);
                let v_hat = v_new.div_scalar_raw(bias_correction2);

                let denom = v_hat.sqrt_raw().add_scalar_raw(self.epsilon);
                let update = m_hat.div_raw(&denom)?.mul_scalar_raw(self.learning_rate);
                let new_param = param.sub_raw(&update)?;
                param.update_data_from(&new_param);

                self.m.insert(param_id, m_new);
                self.v.insert(param_id, v_new);
            }
        }

        Ok(())
    }

    fn lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_new_defaults() {
        let adamw = AdamW::new(0.001);
        assert!((adamw.lr() - 0.001).abs() < f32::EPSILON);
        assert!((adamw.weight_decay - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adamw_with_betas() {
        let adamw = AdamW::new(0.001).with_betas(0.85, 0.99);
        assert!((adamw.beta1 - 0.85).abs() < f32::EPSILON);
        assert!((adamw.beta2 - 0.99).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adamw_with_epsilon() {
        let adamw = AdamW::new(0.001).with_epsilon(1e-7);
        assert!((adamw.epsilon - 1e-7).abs() < 1e-12);
    }

    #[test]
    fn test_adamw_with_weight_decay() {
        let adamw = AdamW::new(0.001).with_weight_decay(0.05);
        assert!((adamw.weight_decay - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adamw_set_lr() {
        let mut adamw = AdamW::new(0.001);
        adamw.set_lr(0.0001);
        assert!((adamw.lr() - 0.0001).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adamw_lr_returns_learning_rate() {
        let adamw = AdamW::new(0.01);
        assert!((adamw.lr() - 0.01).abs() < f32::EPSILON);
    }
}
