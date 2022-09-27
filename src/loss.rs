use std::time::Instant;

use rayon::prelude::*;

use crate::{weights::Weight, activation::ActivationType, network::Network, regularization::l1_regularization};
// use rayon::prelude::*;

pub fn diff_loss(
  f: &dyn Fn(
    &dyn Fn(&Vec<Weight>, &Vec<f32>, ActivationType) -> f32,
    ActivationType,
    &Vec<Weight>,
    &Vec<Vec<f32>>
  ) -> f32,
  nn: &dyn Fn(&Vec<Weight>, &Vec<f32>, ActivationType) -> f32,
  activation: ActivationType,
  vars: &Vec<Weight>,
  x_trues: &Vec<Vec<f32>>,
) -> f32 {
  let h = 0.01;
  return (
    (f(nn, activation, vars, x_trues) + h) - f(nn, activation, vars, x_trues)
  ) / h
}

pub fn partial_diff_loss(
  loss: &dyn Fn(
    &Network,
    &Vec<Vec<f32>>,
    &Vec<Vec<f32>>,
    bool
  ) -> f32,
  w_name: String,
  nn: &Network,
  data_inp: &Vec<Vec<f32>>,
  x_trues: &Vec<Vec<f32>>,
  eps: f32,
  reg_par: f32,
  drop_en: bool,
  is_biases: bool
) -> f32 {
  let mut new_nn = nn.to_owned();
  
  if is_biases {
    new_nn.layers.to_owned().into_iter().for_each(|layer| {
      new_nn.change_bi(layer.name.to_owned(), w_name.to_owned(), eps)
    });
  } else {
    new_nn.layers.to_owned().into_iter().for_each(|layer| {
      new_nn.change_wi(layer.name.to_owned(), w_name.to_owned(), eps)
    });
  }

  let fh = loss(&new_nn, data_inp, x_trues, drop_en) + l1_regularization(&new_nn, reg_par);
  let fo = loss(nn, data_inp, x_trues, drop_en) + l1_regularization(&nn, reg_par);
  
  (fh - fo) / eps
}

pub fn cross_entropy_loss(
  nn: &Network,
  data_inp: &Vec<Vec<f32>>,
  x_trues: &Vec<Vec<f32>>,
  drop_en: bool
) -> f32 {
  let mut sum = 0.0;

  for i in 0..x_trues.len() {
    let nn_out = nn.output(&data_inp[i], drop_en);
    let mean: f32 = (x_trues[i].to_owned().into_iter().enumerate().map(|(j, v)| {
      v * (nn_out[j]).log2() + (1.0 - v) * (1.0 - nn_out[j]).log2()
    })).sum::<f32>() / x_trues[i].len() as f32;

    sum += mean;
  }
  -sum / x_trues.len() as f32
}

pub fn binary_cross_entropy_loss(
  nn: &Network,
  data_inp: &Vec<Vec<f32>>,
  x_trues: &Vec<Vec<f32>>,
  drop_en: bool
) -> f32 {

  let sum = x_trues.to_owned().into_par_iter().enumerate().map(|(i, x_true)| {
    let nn_out = nn.output(&data_inp[i], drop_en);

    (x_true[0] * nn_out[0].log2()) + ((1.0 - x_true[0]) * (1.0 - nn_out[0]).log2())
  }).sum::<f32>();

  -sum / x_trues.len() as f32
}

pub fn loss_mse(
  nn: &Network,
  data_inp: &Vec<Vec<f32>>,
  x_trues: &Vec<Vec<f32>>,
  drop_en: bool
) -> f32 {
  let mut sum = 0.0;

  for i in 0..x_trues.len() {
    let nn_out = nn.output(&data_inp[i], drop_en);

    let mean: f32 = (x_trues[i].to_owned().into_iter().enumerate().map(|(i, v)| {
      // println!("{} - {v}", nn_out[i]);
      (nn_out[i] - v).powf(2.0)
    })).sum::<f32>() / x_trues[i].len() as f32;

    sum += mean;
  }

  sum / x_trues.len() as f32
}