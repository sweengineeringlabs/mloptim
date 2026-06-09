pub(crate) struct OptimizerConfigIngress;

impl OptimizerConfigIngress {
    pub(crate) fn parse_lr(value: &str) -> Option<f32> {
        value.trim().parse::<f32>().ok().filter(|&v| v > 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: parse_lr
    #[test]
    fn test_optimizer_config_ingress_struct_parse_lr_returns_some_for_valid_positive_float() {
        assert!(OptimizerConfigIngress::parse_lr("0.001").is_some());
    }

    /// @covers: parse_lr
    #[test]
    fn test_optimizer_config_ingress_struct_parse_lr_returns_none_for_zero() {
        assert!(OptimizerConfigIngress::parse_lr("0.0").is_none());
    }

    /// @covers: parse_lr
    #[test]
    fn test_optimizer_config_ingress_struct_parse_lr_returns_none_for_invalid_string() {
        assert!(OptimizerConfigIngress::parse_lr("abc").is_none());
    }
}
