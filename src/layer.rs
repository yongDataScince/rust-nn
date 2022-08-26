use std::fmt;

use crate::{
  neuron::Neuron,
  activation::{ActivationType},
  weights::Weight,
};

#[derive(Debug)]
pub struct Layer {
  pub neurons: Vec<Neuron>,
  pub activation: ActivationType,
  pub weights: Vec<Weight>
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      write!(f, "{:?}", self.neurons)
    }
}

impl Layer {
  pub fn perceptron(weights: &Vec<Weight>, n_inp: usize, activation: ActivationType) -> Neuron {
    let start_neurons: Vec<Neuron> = (0..n_inp).into_iter().map(|_| Neuron::rand_neuron(activation)).collect();
  
    let mut vals = vec![];

    for i in 0..n_inp {
      let x = start_neurons[i].value;
      for j in 0..n_inp {
        vals.push( x * weights[j].value );
      }
    }

    Neuron::new(vals.into_iter().sum(), activation)
  }

  pub fn new(n_input: usize, n_output: usize, activation: ActivationType) -> Layer {
    let mut neurons = vec![];
    let mut weights: Vec<Weight> = vec![];
    for _ in 0..n_output  {
        let ws: Vec<Weight> = (0..n_input).into_iter().map(|_| Weight::random_weight()).collect();
        let neuron = Layer::perceptron(&ws, n_input, activation);
        weights.extend(&ws);
        neurons.push(neuron);
    }

    Layer { neurons, activation, weights }
  }
}
