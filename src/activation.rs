use serde::{ Deserialize, Serialize };
use rayon::prelude::*;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Deserialize, Serialize)]
pub enum ActivationType {
  Step,
  Sigmoid,
  Tanh,
  ReLU,
  Softmax,
  No
}

impl ActivationType {
  pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::powf(2.71828, x) )
  }
  pub fn tanh(x: f32) -> f32 {
    return 
      (f32::powf(2.71828, x) - f32::powf(2.71828, -x)) / 
      (f32::powf(2.71828, x) + f32::powf(2.71828, -x))
  }
  pub fn step(x: f32) -> f32 {
    if x >= 0.5 { 1.0 } else { 0.0 }
  }

  pub fn relu(x: f32) -> f32 {
    if x <= 0.0 {
      return 0.0;
    } else {
      return x;
    }
  }

  pub fn softmax(y: &Vec<f32>) -> Vec<f32> {
    let sum: f32 = y.into_par_iter().map(|yi| {
      return 2.71828_f32.powf(*yi); 
    }).sum();
    let out = y.into_iter().map(|yi| {
      2.71828_f32.powf(*yi) / sum 
    }).collect();

    out
  }
}
