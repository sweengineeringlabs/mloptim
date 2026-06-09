use mlautograd::MlResult;
pub use crate::api::optimizer::Optimizer;

pub trait Validator {
    fn validate(&self) -> MlResult<()>;
}
