use crate::{weights::Weight, activation::ActivationType};

pub fn diff_loss(
  f: &dyn Fn(
    &dyn Fn(&Vec<Weight>, &Vec<f64>, ActivationType) -> f64,
    ActivationType,
    &Vec<Weight>,
    &Vec<Vec<f64>>
  ) -> f64,
  nn: &dyn Fn(&Vec<Weight>, &Vec<f64>, ActivationType) -> f64,
  activation: ActivationType,
  vars: &Vec<Weight>,
  x_trues: &Vec<Vec<f64>>,
) -> f64 {
  let h = 0.01;
  return (
    (f(nn, activation, vars, x_trues) + h) - f(nn, activation, vars, x_trues)
  ) / h
}

pub fn partial_diff_loss(
  f: &dyn Fn(
    &dyn Fn(&Vec<Weight>, &Vec<f64>, ActivationType) -> f64,
    &Vec<Weight>,
    ActivationType,
    &Vec<Vec<f64>>,
    &Vec<f64>
  ) -> f64,
  nn: &dyn Fn(&Vec<Weight>, &Vec<f64>, ActivationType) -> f64,
  weights: &Vec<Weight>,
  activation: ActivationType,
  data_inp: &Vec<Vec<f64>>,
  x_trues: &Vec<f64>,
  var_name: String,
  eps: f64
) -> f64 {
  let new_vars: Vec<Weight> = weights.to_owned().iter().map(|w| {
    if w.name == var_name {
      return Weight {
        value: w.value + eps,
        name: var_name.to_owned()
      };
    } else {
      return Weight {
        name: w.name.to_owned(),
        value: w.value
      };
    }
  }).collect();
  ( f(nn, &new_vars, activation, data_inp, x_trues) - f(nn, weights, activation, data_inp, x_trues) ) / eps
}

pub fn loss_mse(
  nn: &dyn Fn(&Vec<Weight>, &Vec<f64>, ActivationType) -> f64,
  weights: &Vec<Weight>,
  activation: ActivationType,
  data_inp: &Vec<Vec<f64>>,
  x_trues: &Vec<f64>
) -> f64 {

  let mut sum: f64 = 0.0;
  for i in 0..x_trues.len() {
    sum += (x_trues[i] - nn(weights, &data_inp[i], activation) ).powf(2.0);
  }

  sum / (x_trues.len() as f64)
}