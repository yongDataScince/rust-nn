use std::{collections::HashMap, time::Instant};
#[warn(unused_imports)]
use std::fmt::{self, Display};
use ndarray::{Array, Dim};
use rand::{distributions::Alphanumeric, Rng, thread_rng};
use ndarray::parallel::prelude::*;
use serde::{ Deserialize, Serialize };
use crate::{
  activation::ActivationType,
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
  pub local_weights: HashMap<String, f64>,
  pub activation: ActivationType
}

impl Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      writeln!(f, "{} -> {}: {:?}", self.n_input, self.n_output, self.activation)
    }
}

impl Layer {
  pub fn output(&self, inputs: Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
    use ActivationType::*;

    let weights: Array<f64, _> = self.local_weights
      .values()
      .map(|v| v.clone())
      .collect::<Vec<f64>>()
      .into();
    
    let weights = weights.into_shared().reshape((self.n_output,  inputs.shape()[1]));
    
    let original_output = (weights.dot(&(inputs.t()))).into_owned();
    let original_output = original_output.t().to_owned();

    match self.activation {
      Step => original_output.to_owned().mapv(|v| if v > 0.5 { 1.0 } else { 0.0 }),
      Sigmoid => original_output.to_owned().mapv(|v| ActivationType::sigmoid(v)),
      Tanh => original_output.to_owned().mapv(|v| ActivationType::tanh(v)),
      ReLU => original_output.to_owned().mapv(|v|  if v >= 0.0 { v } else { 0.0 }),
      Softmax => ActivationType::softmax(&original_output)
    }
  }
}

#[derive(Debug)]
pub struct Out {
  pub error: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Network {
  pub weights: HashMap<String, HashMap<String, f64>>,
  pub layers: HashMap<String, Layer>,
  pub layer_names: Vec<String>,
  vd: HashMap<String, f64>,
  sd: HashMap<String, f64>,
  grads: Vec<Vec<f64>>
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      writeln!(f, "{:?}", self.layers)
    }
}

impl Network {
  pub fn new(
    layers_info: Vec<(&str, usize, usize, ActivationType)>,
  ) -> Network {
    let mut weights: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let mut layers = HashMap::new();
    let mut vd = HashMap::new();
    let mut sd = HashMap::new();
    let mut grads = Vec::new();
    let mut layer_names = Vec::new();

    layers_info.to_owned().into_iter().for_each(|(layer_name, n_input, n_output, activation)| {
      layer_names.push(layer_name.to_string());
      let mut local_weights = HashMap::new();
      let mut curr_grads = Vec::new();
      for _ in 0..(n_input * n_output) {
          let name = random_name();
          let value = thread_rng().gen_range(-100000..=100000) as f64 / 250000.0;
          local_weights.insert(name.to_owned(), value);
          vd.insert(name.to_owned(), 0.0);
          sd.insert(name.to_owned(), 0.0);
          local_weights.insert(name.to_owned(), value); 
          curr_grads.push(0.0);
      }
      let layer = Layer {
        name: layer_name.to_owned(),
        n_input,
        n_output,
        local_weights: local_weights.clone(),
        activation,
      };
      layers.insert(layer_name.to_string().clone(), layer);
      weights.insert(layer_name.to_string(), local_weights);
      grads.push(curr_grads);
    });

    Network {
      weights,
      layers,
      vd,
      sd,
      grads,
      layer_names,
    }
  }
  
  pub fn import_ws(&mut self, inp_ws: Vec<(String, String, f64)>) {
      let mut new_weigths: HashMap<String, HashMap<String, f64>> = HashMap::new();
      
      inp_ws.clone().into_iter().for_each(|(name_l, name_w, value)| {
        new_weigths.entry(name_l)
          .and_modify(|ws| {
            ws.insert(name_w.clone(), value);
          })
          .or_insert(HashMap::from([(name_w, value)]));
      });

      self.weights = new_weigths.clone();

      inp_ws.clone().into_iter().for_each(|(name_l, _, _)| {
          self.layers.entry(name_l.clone()).and_modify(|layer| {
            let mut new_layer = layer.clone();
            let local_weigths = new_weigths.get(&name_l).unwrap();
            new_layer.local_weights = local_weigths.clone();
            
            *layer = new_layer;
          });
      });
  }

  pub fn weigth_count(&self) -> usize {
    self.weights.keys().map(|l| {
      self.weights.get(l).unwrap().len()
    }).sum()
  }

  pub fn change_wi(&mut self, name_l: &String, name_w: &String, sub_value: f64) {
    self.weights.entry(name_l.clone()).and_modify(|ws| {
      ws.entry(name_w.clone()).and_modify(|val| {
        *val -= sub_value
      });
    });

    self.layers.entry(name_l.clone()).and_modify(|layer| {
      layer.local_weights.entry(name_w.clone()).and_modify(|val| {
        *val -= sub_value;
      });
    });
  }

  pub fn output(&self, vals: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
    let mut inp = vals.to_owned();
    for name in self.layer_names.iter() {
      let layer = self.layers.get(name).unwrap();
      let new_inp = layer.to_owned().output(inp.to_owned());
      inp = new_inp;
    }
    inp
  }

  pub fn train_layer(
    &mut self,
    loss: &dyn Fn(
      &Network,
      &Array<f64, Dim<[usize; 2]>>,
      &Array<f64, Dim<[usize; 2]>>,
    ) -> f64,
    values: &Array<f64, Dim<[usize; 2]>>,
    answers: &Array<f64, Dim<[usize; 2]>>,
    lr: f64,
  ) -> Out {
    let betta = 0.9;
    let gamma = 0.999;
    let mut gi = 0;

    for layer_name in &self.layer_names.clone() {
      let weights = self.weights.get(layer_name).unwrap();

      weights.clone().keys().enumerate().for_each(|(i, key)| {
        self.vd.entry(key.to_owned()).and_modify(|val| {
          *val = (val.to_owned() * betta) + (1.0 - betta) * self.grads[gi][i];
        });

        self.sd.entry(key.to_owned()).and_modify(|val| {
          *val = (val.to_owned() * gamma) + (1.0 - gamma) * self.grads[gi][i].powf(2.0);
        });

        let powb = 1.0 - betta.powi((i + 1) as i32);
        let powg = 1.0 - gamma.powi((i + 1) as i32);

        let mt = self.vd.get(&key.to_owned()).unwrap() / powb;
        let vt = self.sd.get(&key.to_owned()).unwrap() / powg;

        let sub_val = lr * mt / (vt.sqrt() + 1e-7);

        let g = partial_diff_loss(loss, &layer_name.clone(), &key.to_owned(), &self, &values, answers, 1e-4);

        self.change_wi(layer_name, &key.to_owned(), -sub_val);

        self.grads[gi][i] = g;
      });

      gi += 1;
    }
    
    let error = loss(&self, &values, &answers);
    Out {
      error
    }
  }

  pub fn weights_to_vec(&self) -> Vec<(String, String, f64)> {
      let mut out_vec = Vec::new();
    
      for layer_name in &self.layer_names {
          let layer_ws = self.weights.get(layer_name).unwrap();
          for (name, value) in layer_ws {
            out_vec.push((layer_name.clone(), name.clone(), *value));
          }
      }

      out_vec
  }
}
