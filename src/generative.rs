use std::{collections::LinkedList, cmp::Ordering};
use rand::Rng;
use crate::{
  utils::rand_name,
  network::Network,
  weights::Weight, stand::Snake
};

pub fn fitness(net: Network, input: Vec<f64>, target: Vec<f64>) -> f64 {
  let net_out = net.layer_output(input);
  let sum: f64 = target.to_owned().into_iter().enumerate().map(|(i, y)| {
      (y - net_out[i]).powf(2.0)
  }).sum();
  sum / target.len() as f64
}

pub fn selection(generation: Vec<Network>, input: Vec<f64>, target: Vec<f64>, delta: f64) -> Vec<Network> {
  let fintess_mean = (generation.to_owned().into_iter().map(|net| {
      fitness(net, input.to_owned(), target.to_owned())
  }).sum::<f64>()) / generation.len() as f64;

  let new_generation = generation.to_owned().into_iter().filter(|net| {
      return fitness(net.to_owned(), input.to_owned(), target.to_owned()) - fintess_mean >= delta
  }).collect();

  new_generation
}

pub fn crossing_over(generation: Vec<Snake>) -> Vec<Snake> {
  let mut new_generation = Vec::new();

  (0..generation.len()).into_iter().for_each(|_| {
      let mut copy_gen = generation.to_owned();
      let minimum_el_1 = copy_gen
        .to_owned()
        .into_iter()
        .min_by(|snake1, snake2| snake1.score.total_cmp(&snake2.score))
        .unwrap();
      let (best_id1, _) = copy_gen.iter().enumerate().find(|(_, snake)| snake.brain.name == minimum_el_1.brain.name).unwrap();
      copy_gen = copy_gen.to_owned().into_iter().filter(|snake| snake.brain.name != minimum_el_1.brain.name).collect();
      
      let minimum_el_2 = copy_gen
        .to_owned()
        .into_iter()
        .min_by(|snake1, snake2| snake1.score.total_cmp(&snake2.score))
        .unwrap();
      let (best_id2, _) = copy_gen.iter().enumerate().find(|(_, snake)| snake.brain.name == minimum_el_2.brain.name).unwrap();

      println!("{}, {}", best_id1, best_id2);
      let parent1 = generation.get(best_id1).unwrap();
      let parent2 = generation.get(best_id2).unwrap();

      let child1_genes_part_1 = parent1.brain.weights.to_owned().as_slice()[0..parent1.brain.weights.len() / 2].to_vec();
      let child1_genes_part_2 = parent2.brain.weights.to_owned().as_slice()[(parent1.brain.weights.len() / 2)..parent2.brain.weights.len()].to_vec();
      let child1_ws = [&child1_genes_part_1[..], &child1_genes_part_2[..],].concat();
      
      let child1 = Snake {
        brain: Network::from_flatten(
          child1_ws, 
          rand_name(), 
          parent1.brain.in_weights.len(), 
          parent1.brain.n_hidden, 
          parent1.brain.out_weights.len() / parent1.brain.n_hidden, 
          parent1.brain.activation,
          parent1.brain.out_activation
        ),
        speed: 0.3,
        head: (0, 0),
        body: LinkedList::new(),
        dir: (1, 0),
        min_dist: 10000.0,
        score: 0.0,
        game_over: false
      };

      let child2_genes_part_1 = parent2.brain.weights.to_owned().as_slice()[0..parent1.brain.weights.len() / 2].to_vec();
      let child2_genes_part_2 = parent1.brain.weights.to_owned().as_slice()[(parent1.brain.weights.len() / 2)..parent1.brain.weights.len()].to_vec();
      let child2_ws = [&child2_genes_part_1[..], &child2_genes_part_2[..],].concat();

      let child2 = Snake {
        brain: Network::from_flatten(
          child2_ws, 
          rand_name(),
          parent1.brain.in_weights.len(), 
          parent1.brain.n_hidden, 
          parent1.brain.out_weights.len() / parent1.brain.n_hidden, 
          parent1.brain.activation,
          parent1.brain.out_activation
      ),
      speed: 0.3,
      head: (0, 0),
      body: LinkedList::new(),
      dir: (1, 0),
      min_dist: 10000.0,
      score: 0.0,
      game_over: false
      };

      new_generation.push(child1);
      new_generation.push(child2);
  });

  new_generation
}

pub fn mutation(generation: Vec<Snake>, p: f64) -> Vec<Snake> {
  let mut new_generation = Vec::new();

  generation.to_owned().into_iter().for_each(|net| {
      let mut new_net = net.to_owned();
      if rand::thread_rng().gen_range(0..=10) as f64 / 10.0 <= p {
          let mut new_ws = net.brain.weights.to_owned();
          let rw_id = rand::thread_rng().gen_range(0..net.brain.weights.len());
          
          new_ws[rw_id] = Weight {
            value: new_ws[rw_id].value + rand::thread_rng().gen_range(-100..=1000) as f64 / 1000.0,
            name: new_ws[rw_id].name.to_owned()
          };
        new_net.brain.weights = new_ws;
      } 

      new_generation.push(new_net);
  });

  new_generation
}

pub fn choise_best(generation: Vec<Network>, inps: Vec<f64>, target: Vec<f64>) -> Network {
  let mut min = 1000000000000.0;
  let mut min_idx = generation.len() - 1;
  for i in 0..generation.len() {
      let f = fitness(generation[i].to_owned(), inps.to_owned(), target.to_owned());
      if f < min {
          min = f;
          min_idx = i;
      }
  }

  return generation[min_idx].to_owned();
}
