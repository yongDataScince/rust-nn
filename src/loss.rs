use std::time::Instant;

use rayon::prelude::*;
use crate::{weights::Weight, activation::ActivationType, network::Network};

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
  loss: &dyn Fn(
    &Network,
    &Vec<Vec<f64>>,
    &Vec<Vec<f64>>,
  ) -> f64,
  l_name: &String,
  w_name: &String,
  nn: &Network,
  data_inp: &Vec<Vec<f64>>,
  x_trues: &Vec<Vec<f64>>,
  eps: f64,
) -> f64 {
  let mut new_nn = nn.to_owned();

  let start = Instant::now();
  new_nn.change_wi(l_name, w_name, eps);
  let duration = start.elapsed();
  // println!("    Change weight in part. diff: {:?}", duration);

  let start = Instant::now();
  let fh = loss(&new_nn, data_inp, x_trues);
  let duration = start.elapsed();
  // println!("    Calc fh in part. diff: {:?}", duration);
  let fo = loss(nn, data_inp, x_trues);
  
  (fh - fo) / eps
}

pub fn cross_entropy_loss(
  nn: &Network,
  data_inp: &Vec<Vec<f64>>,
  x_trues: &Vec<Vec<f64>>,
) -> f64 {
  let mut sum = 0.0;

  for i in 0..x_trues.len() {
    let nn_out = nn.output(&data_inp[i]);
    let mean: f64 = (x_trues[i].to_owned().into_iter().enumerate().map(|(j, v)| {
      v * (nn_out[j]).log2() + (1.0 - v) * (1.0 - nn_out[j]).log2()
    })).sum::<f64>() / x_trues[i].len() as f64;

    sum += mean;
  }
  -sum / x_trues.len() as f64
}

pub fn binary_cross_entropy_loss(
  nn: &Network,
  data_inp: &Vec<Vec<f64>>,
  x_trues: &Vec<Vec<f64>>,
) -> f64 {

  let sum = x_trues.to_owned().into_par_iter().enumerate().map(|(i, x_true)| {
    let nn_out = nn.output(&data_inp[i]);

    (x_true[0] * nn_out[0].log2()) + ((1.0 - x_true[0]) * (1.0 - nn_out[0]).log2())
  }).sum::<f64>();

  -sum / x_trues.len() as f64
}



pub fn loss_mse(
  nn: &Network,
  data_inp: &Vec<Vec<f64>>,
  x_trues: &Vec<Vec<f64>>,
) -> f64 {
  let mut sum = 0.0;

  for i in 0..x_trues.len() {
    let nn_out = nn.output(&data_inp[i]);

    let mean: f64 = (x_trues[i].to_owned().into_iter().enumerate().map(|(i, v)| {
      (nn_out[i] - v).powf(2.0)
    })).sum::<f64>() / x_trues[i].len() as f64;

    sum += mean;
  }

  sum / x_trues.len() as f64
}