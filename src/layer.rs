use std::fmt;
use rand::Rng;
use crate::{
  activation::{ActivationType},
  weights::Weight, loss::{loss_mse, partial_diff_loss},
};

#[derive(Debug)]
pub struct Out {
  pub error: f64,
  pub weights: Vec<Weight>
}

#[derive(Debug, Clone)]
pub struct Layer {
  pub activation: ActivationType,
  pub out_activation: ActivationType,
  pub weights: Vec<Weight>,
  pub in_weights: Vec<Weight>,
  pub hidden_weights: Vec<Weight>,
  pub n_hidden: usize,
  pub n_out: usize,
  pub out_weights: Vec<Weight>,
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "{:?}", self.weights)
    }
}

impl Layer {
  pub fn new(n_inp: usize, n_hidden: usize, out_dim: usize, activation: ActivationType, out_activation: ActivationType) -> Layer {
    let mut weights = Vec::new();
    let mut idx = 0;
    let in_weights: Vec<Weight> = (0..n_inp).into_iter().map(|_| {
      let w = Weight::random_weight(format!("w{}", idx));
      idx += 1;
      weights.push(w.to_owned());
      w
    }).collect();
    let hidden_weights: Vec<Weight> = (0..n_inp * n_hidden).into_iter().map(|_| {
      let w = Weight::random_weight(format!("w{}", idx));
      idx += 1;
      weights.push(w.to_owned());
      w
    }).collect();
    
    let out_weights: Vec<Weight> = (0..out_dim * n_hidden).into_iter().map(|_| {
      let w = Weight::random_weight(format!("w{}", idx));
      idx += 1;
      weights.push(w.to_owned());
      w
    }).collect();

    Layer {
      activation,
      in_weights,
      hidden_weights,
      out_activation,
      out_weights,
      n_hidden,
      n_out: out_dim,
      weights
    }
  }

  pub fn layer_output(&self, vals: Vec<f64>) -> Vec<f64> {
    use ActivationType::*;

    let first_out: Vec<f64> = self.in_weights.iter().enumerate().map(|(i, w)| w.value * vals[i]).collect();

    let hidden_out: Vec<f64> = self.hidden_weights.iter().map(|w| {
      let mut sum = 0.0;
      first_out.iter().for_each(|val| {
        sum += match self.activation {
          Step => if w.value * val > 0.5 { 1.0 } else { 0.0 },
          Sigmoid => ActivationType::sigmoid(w.value * val),
          Tanh => ActivationType::tanh(w.value * val),
          ReLU => ActivationType::relu(w.value * val),
          _ => 0.0
        };
      });
      return sum;
    }).collect();

    let out: Vec<f64> = self.out_weights.chunks(self.n_hidden).map(|chunk| {
      let sum = chunk.iter().map(|w| {
        let mut sum: f64 = 0.0;
        hidden_out.iter().for_each(|val| {
          let out = match self.activation {
            Step => if w.value * val > 0.5 { 1.0 } else { 0.0 },
            Sigmoid => ActivationType::sigmoid(w.value * val),
            Tanh => ActivationType::tanh(w.value * val),
            ReLU => ActivationType::relu(w.value * val),
            _ => 0.0
          };
          sum += out;
        });
        match self.activation {
          Step => if sum > 0.5 { 1.0 } else { 0.0 },
          Sigmoid => ActivationType::sigmoid(sum),
          Tanh => ActivationType::tanh(sum),
          ReLU => ActivationType::relu(sum),
          _ => 0.0
        }
      }).sum::<f64>();
      match self.activation {
        Step => if sum > 0.5 { 1.0 } else { 0.0 },
        Sigmoid => ActivationType::sigmoid(sum),
        Tanh => ActivationType::tanh(sum),
        ReLU => ActivationType::relu(sum),
        _ => 0.0
      }
    }).collect();

    match self.out_activation {
        Step => out.iter().map(|v| if v > &0.5 { 1.0 } else { 0.0 }).collect(),
        Sigmoid => out.iter().map(|v| ActivationType::sigmoid(*v)).collect(),
        Tanh => out.iter().map(|v| ActivationType::tanh(*v)).collect(),
        ReLU => out.iter().map(|v| ActivationType::relu(*v)).collect(),
        Softmax => ActivationType::softmax(&out),
    }
  }

  pub fn call(
    in_weights: &Vec<Weight>,
    hidden_weights: &Vec<Weight>,
    out_weights: &Vec<Weight>,
    vals: &Vec<f64>,
    n_hidden: usize,
    activation: ActivationType,
    out_activation: ActivationType,
  ) -> Vec<f64> {
    use ActivationType::*;
    let first_out: Vec<f64> = in_weights.iter().enumerate().map(|(i, w)| w.value * vals[i]).collect();

    let hidden_out: Vec<f64> = hidden_weights.iter().map(|w| {
      let mut sum = 0.0;
      first_out.iter().for_each(|val| {
        sum += match activation {
          Step => if w.value * val > 0.5 { 1.0 } else { 0.0 },
          Sigmoid => ActivationType::sigmoid(w.value * val),
          Tanh => ActivationType::tanh(w.value * val),
          ReLU => ActivationType::relu(w.value * val),
        _ => 0.0
        };
      });
      return sum;
    }).collect();

    let out: Vec<f64> = out_weights.chunks(n_hidden).map(|chunk| {
      let sum = chunk.iter().map(|w| {
        let mut sum: f64 = 0.0;
        hidden_out.iter().for_each(|val| {
          let out = match activation {
            Step => if w.value * val > 0.5 { 1.0 } else { 0.0 },
            Sigmoid => ActivationType::sigmoid(w.value * val),
            Tanh => ActivationType::tanh(w.value * val),
            ReLU => ActivationType::relu(w.value * val),
            _ => 0.0
          };
          sum += out;
        });
        match activation {
          Step => if sum > 0.5 { 1.0 } else { 0.0 },
          Sigmoid => ActivationType::sigmoid(sum),
          Tanh => ActivationType::tanh(sum),
          ReLU => ActivationType::relu(sum),
          _ => 0.0
        }
      }).sum::<f64>();
      match activation {
        Step => if sum > 0.5 { 1.0 } else { 0.0 },
        Sigmoid => ActivationType::sigmoid(sum),
        Tanh => ActivationType::tanh(sum),
        ReLU => ActivationType::relu(sum),
        _ => 0.0
      }
    }).collect();

    match out_activation {
      Step => out.iter().map(|v| if v > &0.5 { 1.0 } else { 0.0 }).collect(),
      Sigmoid => out.iter().map(|v| ActivationType::sigmoid(*v)).collect(),
      Tanh => out.iter().map(|v| ActivationType::tanh(*v)).collect(),
      ReLU => out.iter().map(|v| ActivationType::relu(*v)).collect(),
      Softmax => ActivationType::softmax(&out)
    }
  }

  pub fn train_layer(&mut self, values: Vec<Vec<f64>>, answers: &Vec<Vec<f64>>, lr: f64) -> Out {
    let mut trained: Vec<Weight> = Vec::new();

    while trained.len() < self.weights.len() / 2 {
      let ri = rand::thread_rng().gen_range(0..self.weights.to_owned().len());
      if !trained.contains(&self.weights[ri].to_owned()) {
        self.weights[ri] = Weight {
          value: self.weights[ri].value - lr * partial_diff_loss(
            &loss_mse,
            &Layer::call,
            &self.weights,
            &self.in_weights,
            &self.hidden_weights,
            &self.out_weights,
            self.activation,
            self.n_hidden,
            self.out_activation,
            &values,
            &answers,
            self.weights[ri].name.to_owned(),
            0.0001
          ),
          name: self.weights[ri].name.to_owned()
        };
        match self.in_weights.iter().position(|w| w.name == self.weights[ri].name) {
            Some(id) => { self.in_weights[id] = self.weights[ri].to_owned() },
            None => (),
        };
        match self.hidden_weights.iter().position(|w| w.name == self.weights[ri].name) {
          Some(id) => { self.hidden_weights[id] = self.weights[ri].to_owned() },
          None => (),
        };
        match self.out_weights.iter().position(|w| w.name == self.weights[ri].name) {
          Some(id) => { self.out_weights[id] = self.weights[ri].to_owned() },
          None => (),
        };

        trained.push(self.weights[ri].to_owned());
      }
    }
    let error = loss_mse(
      &Layer::call,
      &self.in_weights,
      &self.hidden_weights,
      &self.out_weights,
      &values,
      answers,
      self.n_hidden,
      self.activation,
      self.out_activation
    );
    Out {
      error,
      weights: self.weights.to_owned(),
    }
  }
}
