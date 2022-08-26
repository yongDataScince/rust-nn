use thiserror::Error;

#[derive(Debug, Error)]
pub enum NNErrors {
  #[error("Activation not eqal")]
  NotEqActivation
}