use mlautograd::{Tensor, tape};

/// Clip gradients by global L2 norm.
///
/// Computes the total L2 norm across all parameter gradients currently stored
/// on the tape. If the total norm exceeds `max_norm`, every gradient is
/// scaled down by `max_norm / total_norm` and written back via `tape::set_grad`.
///
/// Returns the *original* (unclipped) total norm so callers can log it.
pub fn clip_grad_norm(params: &[&Tensor], max_norm: f32) -> f32 {
    let mut total_norm_sq: f64 = 0.0;
    let mut grads: Vec<Option<Tensor>> = Vec::with_capacity(params.len());

    for param in params {
        let g = tape::grad(param);
        if let Some(ref grad) = g {
            let grad_data = grad.to_vec();
            let sq_sum: f64 = grad_data.iter().map(|&x| (x as f64) * (x as f64)).sum();
            total_norm_sq += sq_sum;
        }
        grads.push(g);
    }

    let total_norm = (total_norm_sq as f32).sqrt();

    if total_norm > max_norm {
        let clip_coef = max_norm / total_norm;
        for (param, grad_opt) in params.iter().zip(grads.into_iter()) {
            if let Some(grad) = grad_opt {
                let clipped = grad.mul_scalar_raw(clip_coef);
                tape::set_grad(param, clipped);
            }
        }
    }

    total_norm
}

/// Clip gradients by value.
///
/// Clamps every element of every parameter gradient to the range
/// `[-clip_value, clip_value]` and writes the result back to the tape.
pub fn clip_grad_value(params: &[&Tensor], clip_value: f32) {
    let clip_value = clip_value.abs();

    for param in params {
        if let Some(grad) = tape::grad(param) {
            let data = grad.to_vec();
            let clamped: Vec<f32> = data
                .into_iter()
                .map(|x| x.clamp(-clip_value, clip_value))
                .collect();
            let shape = grad.shape().to_vec();
            let clipped =
                Tensor::from_vec(clamped, shape).expect("clip_grad_value: shape mismatch");
            tape::set_grad(param, clipped);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_grad_norm_returns_zero_when_no_grads() {
        tape::clear_tape();
        let t = Tensor::zeros(vec![3]);
        let norm = clip_grad_norm(&[&t], 1.0);
        assert!(norm.abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_value_does_not_panic_without_grads() {
        tape::clear_tape();
        let t = Tensor::zeros(vec![3]);
        clip_grad_value(&[&t], 0.5);
    }
}
