#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ActivationType {
  Step,
  Sigmoid,
  Tanh
}

impl ActivationType {
  pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::powf(2.71828, x) )
  }
  pub fn tanh(x: f64) -> f64 {
    return 
      (f64::powf(2.71828, x) - f64::powf(2.71828, -x)) / 
      (f64::powf(2.71828, x) + f64::powf(2.71828, -x))
  }
  pub fn step(x: f64) -> f64 {
    if x >= 0.5 { 1.0 } else { 0.0 }
  }
}
