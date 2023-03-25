use std::time::Instant;

use ndarray::{Array, Dim};
use ndarray::parallel::prelude::*;
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
    &Array<f64, Dim<[usize; 2]>>,
    &Array<f64, Dim<[usize; 2]>>,
  ) -> f64,
  l_name: &String,
  w_name: &String,
  nn: &Network,
  data_inp: &Array<f64, Dim<[usize; 2]>>,
  x_trues: &Array<f64, Dim<[usize; 2]>>,
  eps: f64,
) -> f64 {
  let mut new_nn = nn.to_owned();

  let start = Instant::now();
  new_nn.change_wi(l_name, w_name, eps);
  let duration = start.elapsed();
  println!("    Change weight in part. diff: {:?}", duration);

  let start = Instant::now();
  let fh = loss(&new_nn, data_inp, x_trues);
  let duration = start.elapsed();
  println!("    Calc fh in part. diff: {:?}", duration);
  let fo = loss(nn, data_inp, x_trues);
  
  (fh - fo) / eps
}

pub fn cross_entropy_loss(
  nn: &Network,
  data_inp: &Array<f64, Dim<[usize; 2]>>,
  x_trues: &Array<f64, Dim<[usize; 2]>>,
) -> f64 {
  let nn_out = nn.output(&data_inp);

  let mut log_out = nn_out.clone();
  log_out.par_mapv_inplace(|v| v.log2());

  let mut div_log = 1.0 - nn_out.clone();
  div_log.par_mapv_inplace(|v| v.log2());

  let out = x_trues * log_out + (1.0 - x_trues) * div_log.clone();

  -out.mean().unwrap()
}

pub fn binary_cross_entropy_loss(
  nn: &Network,
  data_inp: &Array<f64, Dim<[usize; 2]>>,
  x_trues: &Array<f64, Dim<[usize; 2]>>,
) -> f64 {
  let nn_out = nn.output(&data_inp);

  let out = (x_trues * nn_out.clone().mapv_into(|v| v.log2())) + ((1.0 - x_trues) * (1.0 - nn_out.mapv_into(|v| v.log2())));

  out.mean().unwrap()
}

pub fn loss_mse(
  nn: &Network,
  data_inp: &Array<f64, Dim<[usize; 2]>>,
  x_trues: &Array<f64, Dim<[usize; 2]>>,
) -> f64 {
  let nn_out = nn.output(&data_inp);
  let mut dif = nn_out - x_trues;
  dif.par_mapv_inplace(|v| v.powf(2.0));

  dif.mean().unwrap()
}