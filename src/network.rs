use std::{collections::HashMap, time::Instant};
#[warn(unused_imports)]
use std::fmt::{self, Display};
use rand::{distributions::Alphanumeric, Rng, thread_rng};
use rayon::prelude::*;
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
  pub fn output(&self, inputs: Vec<f64>) -> Vec<f64> {
    use ActivationType::*;
    let original_output: Vec<f64> = self.local_weights.values().collect::<Vec<&f64>>().chunks(self.n_input).map(|chunk| {
      inputs.clone().into_iter().enumerate().map(|(i, w)| {
        chunk[i] * w
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
          let value = thread_rng().gen_range(-10000..=10000) as f64 / 10000.0;
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

  pub fn output(&self, vals: &Vec<f64>) -> Vec<f64> {
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
      &Vec<Vec<f64>>,
      &Vec<Vec<f64>>,
    ) -> f64,
    values: Vec<Vec<f64>>,
    answers: &Vec<Vec<f64>>,
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

        let start = Instant::now();
        let g = partial_diff_loss(loss, &layer_name.clone(), &key.to_owned(), &self, &values, answers, 1e-4);
        let duration = start.elapsed();
        // println!("  Calculate part. diff: {:?}", duration);

        let start = Instant::now();
        self.change_wi(layer_name, &key.to_owned(), -sub_val);
        let duration = start.elapsed();
        // println!("  Change weight: {:?}", duration);

        self.grads[gi][i] = g;
      });

      gi += 1;
    }
    
    let error = loss(&self, &values[0..values.len()].to_vec(), &answers[0..values.len()].to_vec());
    Out {
      error
    }
  }
}
