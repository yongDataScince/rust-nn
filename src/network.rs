use std::{collections::HashMap, time::Instant};
#[warn(unused_imports)]
use std::{fmt::{self, Display}, time::Duration};
use rand::{distributions::Alphanumeric, Rng};
use rayon::prelude::*;
use serde::{ Deserialize, Serialize };
use crate::{
  activation::ActivationType,
  weights::Weight,
  loss::partial_diff_loss,
};

fn random_name() -> String {
  rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(16)
        .map(char::from)
        .collect()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Layer {
  pub name: String,
  pub n_input: usize,
  pub n_output: usize,
  pub local_weights: Vec<Weight>,
  pub activation: ActivationType
}

impl Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      writeln!(f, "{} -> {}: {:?}", self.n_input, self.n_output, self.activation)
    }
}

impl Layer {
  pub fn output(self, inputs: Vec<f64>) -> Vec<f64> {
    use ActivationType::*;
    let original_output: Vec<f64> = inputs.to_owned().into_par_iter().map(|inp| {
      self.local_weights.to_owned().into_par_iter().map(|w| {
        w.value * inp
      }).sum::<f64>()
    }).collect();

    match self.activation {
      Step => original_output.to_owned().into_par_iter().map(|v| if v > 0.5 { 1.0 } else { 0.0 }).collect(),
      Sigmoid => original_output.to_owned().into_par_iter().map(|v| ActivationType::sigmoid(v)).collect(),
      Tanh => original_output.to_owned().into_par_iter().map(|v| ActivationType::tanh(v)).collect(),
      ReLU => original_output.to_owned().into_par_iter().map(|v|  if v >= 0.0 { v } else { 0.0 }).collect(),
      Softmax => ActivationType::softmax(&original_output)
    }
  }
}

#[derive(Debug)]
pub struct Out {
  pub error: f64,
  pub weights: Vec<Weight>
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Network {
  pub weights: Vec<Weight>,
  pub layers: Vec<Layer>,
  vd: HashMap<String, f64>,
  sd: HashMap<String, f64>,
  grads: Vec<f64>
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      writeln!(f, "{:?}", self.layers)
    }
}

impl Network {
  pub fn new(
    layers_info: Vec<(String, usize, usize, ActivationType)>,
  ) -> Network {
    let mut weights: Vec<Weight> = Vec::new();
    
    let mut vd = HashMap::new();
    let mut sd = HashMap::new();
    let mut grads = Vec::new();

    let layers: Vec<Layer> = layers_info.to_owned().into_iter().enumerate().map(|(i, (name, n_input, n_output, activation))| {
      let mut local_weights: Vec<Weight> = Vec::new();
      for _ in 0..(n_input * n_output) {
        let w = Weight::random_weight(random_name());
        weights.push(w.to_owned());
        vd.insert(w.name.to_owned(), 0.0);
        sd.insert(w.name.to_owned(), 0.0);
        local_weights.push(w.to_owned());
      }
      weights = weights.to_owned().into_iter().map(|w| {
        let new_val = w.to_owned().value * ((i as f64).sqrt() / (local_weights.len() as f64).sqrt());
        Weight {
          value: new_val,
          name: w.name.to_owned()
        }
      }).collect();
      return Layer { name, n_input, n_output, local_weights, activation }
    }).collect();

    (0..weights.len()).for_each(|_| grads.push(0.0));

    Network {
      weights,
      layers,
      vd,
      sd,
      grads,
    }
  }
  
  pub fn change_wi(&mut self, name_l: String, name_w: String, sub_value: f64) {
    let (layer_id, mut curr_layer) = self.layers.to_owned().into_par_iter().enumerate().find_any(|(_, layer)| layer.name == name_l.to_owned()).unwrap();
    let (w_id, mut curr_w) = self.weights.to_owned().into_par_iter().enumerate().find_any(|(_, w)| w.name == name_w.to_owned()).unwrap();

    curr_layer.local_weights = curr_layer.local_weights.into_iter().map(|w| {
      if w.name == name_w.to_owned() {
        curr_w = Weight {
          name: name_w.to_owned(),
          value: w.value + sub_value.to_owned()
        };
        return Weight {
          name: name_w.to_owned(),
          value: w.value + sub_value.to_owned()
        }
      } else {
        w
      }
    }).collect();

    self.layers[layer_id] = curr_layer;
    self.weights[w_id] = curr_w;
  }

  pub fn output(&self, vals: &Vec<f64>) -> Vec<f64> {
    let mut inp = vals.to_owned();
    for layer in self.layers.iter() {
      let new_inp = layer.to_owned().output(inp.to_owned());
      inp = new_inp;
    }
    inp
  }
  pub fn train_layer(
    &mut self,
    loss: &dyn Fn(
      &Network,
      &Vec<Vec<f64>>,
      &Vec<Vec<f64>>,
    ) -> f64,
    values: Vec<Vec<f64>>,
    answers: &Vec<Vec<f64>>,
    lr: f64,
  ) -> Out {
    let betta = 0.9;
    let gamma = 0.999;

    let mut i = 0;
    while i < self.weights.len() {
      self.vd.entry(self.weights[i].name.to_owned()).and_modify(|val| {
        *val = (val.to_owned() * betta) + (1.0 - betta) * self.grads[i];
      });

      self.sd.entry(self.weights[i].name.to_owned()).and_modify(|val| {
        *val = (val.to_owned() * gamma) + (1.0 - gamma) * self.grads[i].powf(2.0);
      });

      let powb = 1.0 - betta.powi((i + 1) as i32);
      let powg = 1.0 - gamma.powi((i + 1) as i32);

      // println!("powb: {powb}");
      // println!("powg: {powg}");

      let mt = self.vd.get(&self.weights[i].name.to_owned()).unwrap() / powb;
      let vt = self.sd.get(&self.weights[i].name.to_owned()).unwrap() / powg;

      // println!("mt: {} / {powb} = {}", self.vd.get(&self.weights[i].name.to_owned()).unwrap(), mt);
      // println!("vt: {} / {powg} = {}", self.sd.get(&self.weights[i].name.to_owned()).unwrap(), vt);
      // println!("sub_val: {} / {} = {}\n", lr * mt, (vt.sqrt() + 1e-7), lr * mt / (vt.sqrt() + 1e-7));
      let sub_val = lr * mt / (vt.sqrt() + 1e-7);

      let g = partial_diff_loss(loss, self.weights[i].name.to_owned(), &self, &values, answers, 1e-7);

      self.layers.to_owned().iter().for_each(|layer| {
        self.change_wi(layer.name.to_owned(), self.weights[i].name.to_owned(), -sub_val)
      });

      self.grads[i] = g;
      i += 1;
    }
    let error = loss(&self, &values[0..values.len()].to_vec(), &answers[0..values.len()].to_vec());
    Out {
      error,
      weights: self.weights.to_owned(),
    }
  }
}
