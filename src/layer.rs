use std::fmt;

use rand::Rng;

use crate::{
  activation::{ActivationType},
  weights::Weight, loss::{loss_mse, partial_diff_loss},
};

#[derive(Debug)]
pub struct Layer {
  pub activation: ActivationType,
  pub weights: Vec<Weight>
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "{:?}", self.weights)
    }
}

impl Layer {
  pub fn new(n_inp: usize, activation: ActivationType) -> Layer {
    let weights: Vec<Weight> = (0..n_inp).into_iter().map(|i| Weight::random_weight(format!("w{}", i))).collect();

    Layer {
      activation,
      weights
    }
  }

  pub fn layer_output(&self, vals: Vec<f64>) -> f64 {
    use ActivationType::*;
    let value = self.weights.iter().enumerate().map(|(i, w)| w.value * vals[i]).sum();
    match self.activation {
      Step => if value > 0.5 { 1.0 } else { 0.0 },
      Sigmoid => ActivationType::sigmoid(value),
      Tanh => ActivationType::tanh(value)
    }
  }

  pub fn call(weights: &Vec<Weight>, vals: &Vec<f64>, activation: ActivationType) -> f64 {
    use ActivationType::*;
    let value = weights.iter().enumerate().map(|(i, w)| w.value * vals[i]).sum();
    match activation {
      Step => if value > 0.5 { 1.0 } else { 0.0 },
      Sigmoid => ActivationType::sigmoid(value),
      Tanh => ActivationType::tanh(value)
    }
  }

  pub fn train_layer(&mut self, values: Vec<Vec<f64>>, answers: &Vec<f64>, lr: f64) {
    let mut trained: Vec<Weight> = Vec::new();
    while trained.len() < self.weights.len() {
      let ri = rand::thread_rng().gen_range(0..self.weights.to_owned().len());

      if !trained.contains(&self.weights[ri].to_owned()) {
        self.weights[ri] = Weight {
          value: self.weights[ri].value - lr * partial_diff_loss(
            &loss_mse,
            &Layer::call,
            &self.weights,
            self.activation,
            &values,
            answers,
            self.weights[ri].name.to_owned(),
            0.0001
          ),
          name: self.weights[ri].name.to_owned()
        };
        let error = loss_mse(&Layer::call, &self.weights, self.activation, &values, answers);
        println!("error: {}", error);
        trained.push(self.weights[ri].to_owned());
      }
    }
  }
}
