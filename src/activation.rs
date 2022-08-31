#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ActivationType {
  Step,
  Sigmoid,
  Tanh,
  ReLU,
  Softmax
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

  pub fn relu(x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    } else {
      return x;
    }
  }

  pub fn softmax(y: &Vec<f64>) -> Vec<f64> { 
    let sum: f64 = y.into_iter().map(|yi| {
      return 2.71828_f64.powf(*yi); 
    }).sum();

    y.into_iter().map(|yi| {
      2.71828_f64.powf(*yi) / sum 
    }).collect()
  }
}
