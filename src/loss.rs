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
    &dyn Fn(&Vec<Weight>, &Vec<Weight>, &Vec<Weight>, &Vec<f64>, usize, ActivationType, ActivationType) -> Vec<f64>,
    &Vec<Weight>,
    &Vec<Weight>,
    &Vec<Weight>,
    &Vec<Vec<f64>>,
    &Vec<Vec<f64>>,
    usize,
    ActivationType,
    ActivationType,
  ) -> f64,
  nn: &dyn Fn(&Vec<Weight>, &Vec<Weight>, &Vec<Weight>, &Vec<f64>, usize, ActivationType, ActivationType) -> Vec<f64>,
  all_weights: &Vec<Weight>, 
  in_weights: &Vec<Weight>,
  hidden_weights: &Vec<Weight>,
  out_weights: &Vec<Weight>,
  activation: ActivationType,
  n_hidden: usize,
  out_activation: ActivationType,
  data_inp: &Vec<Vec<f64>>,
  x_trues: &Vec<Vec<f64>>,
  var_name: String,
  eps: f64
) -> f64 {
  let all_weights: Vec<Weight> = all_weights.to_owned().iter().map(|w| {
    if w.name == var_name {
      println!("{}", var_name);
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

  let new_in_w: Vec<Weight> = all_weights.to_owned().into_iter().filter(|w| in_weights.contains(w)).collect();
  let new_hid_w: Vec<Weight> = all_weights.to_owned().into_iter().filter(|w| hidden_weights.contains(w)).collect();
  let new_out_w: Vec<Weight> = all_weights.to_owned().into_iter().filter(|w| out_weights.contains(w)).collect();

  ( f(
      nn,
      &new_in_w,
      &new_hid_w,
      &new_out_w,
      data_inp,
      x_trues,
      n_hidden,
      activation,
      out_activation
    ) - f(
      nn,
      in_weights,
      hidden_weights,
      out_weights,
      data_inp,
      x_trues,
      n_hidden,
      activation,
      out_activation
    )) / eps
}

pub fn cross_entropy_loss(
  nn: &dyn Fn(&Vec<Weight>, &Vec<Weight>, &Vec<Weight>, &Vec<f64>, usize, ActivationType, ActivationType) -> Vec<f64>,
  in_weights: &Vec<Weight>,
  hidden_weights: &Vec<Weight>,
  out_weights: &Vec<Weight>,
  data_inp: &Vec<Vec<f64>>,
  x_trues: &Vec<Vec<f64>>,
  n_hidden: usize,
  activation: ActivationType,
  out_activation: ActivationType,
) -> f64 {
  let mut sum = 0.0;

  for i in 0..x_trues.len() {
    let nn_out = nn(
      in_weights,
      hidden_weights,
      out_weights,
      &data_inp[i],
      n_hidden,
      activation,
      out_activation
    );
    println!("{:?}", nn_out);

    let mean: f64 = (x_trues[i].to_owned().into_iter().enumerate().map(|(i, v)| {
      v * (nn_out[i]).log2() + (1.0 - v) * (1.0 - nn_out[i]).log2()
    })).sum::<f64>() / x_trues[i].len() as f64;

    sum += mean;
  }
  -sum / x_trues.len() as f64
}

pub fn loss_mse(
  nn: &dyn Fn(&Vec<Weight>, &Vec<Weight>, &Vec<Weight>, &Vec<f64>, usize, ActivationType, ActivationType) -> Vec<f64>,
  in_weights: &Vec<Weight>,
  hidden_weights: &Vec<Weight>,
  out_weights: &Vec<Weight>,
  data_inp: &Vec<Vec<f64>>,
  x_trues: &Vec<Vec<f64>>,
  n_hidden: usize,
  activation: ActivationType,
  out_activation: ActivationType,
) -> f64 {
  let mut sum: f64 = 0.0;
  for i in 0..x_trues.len() {
    let nn_out = nn(
      in_weights,
      hidden_weights,
      out_weights,
      &data_inp[i],
      n_hidden,
      activation,
      out_activation
    );

    if nn_out.len() == 1 {
      sum += (x_trues[i][0] - nn_out[0]).powf(2.0); 
    } else {
      sum += nn_out.iter().enumerate().map(|(j, out)| {
        (x_trues[i][j] - out).powf(2.0)
      }).sum::<f64>() / nn_out.len() as f64;
    }
  }

  sum / (x_trues.len() as f64)
}