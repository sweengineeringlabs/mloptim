pub(crate) struct OptimizerStateEgress;

impl OptimizerStateEgress {
    pub(crate) fn format_lr(lr: f32, step: usize) -> String {
        format!("step={step},lr={lr:.6}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: format_lr
    #[test]
    fn test_optimizer_state_egress_struct_format_lr_contains_step_and_lr() {
        let s = OptimizerStateEgress::format_lr(0.001, 10);
        assert!(s.contains("step=10"));
        assert!(s.contains("lr="));
    }
}
