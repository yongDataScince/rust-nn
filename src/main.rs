pub mod loss;
pub mod activation;
pub mod errors;
pub mod network;
pub mod weights;
pub mod loader;
pub mod utils;

use {rand::Rng, rand::distributions::Alphanumeric};

use crate::{
    network::Network,
    activation::ActivationType,
};

pub struct GenResult {
    name: String,
    value: Vec<f64>
}

fn rand_name() -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(7)
        .map(char::from)
        .collect()
}

fn create_generation(
    n: usize,
    n_inp: usize,
    n_hidden: usize,
    out_dim: usize,
    activation: ActivationType,
    out_activation: ActivationType
) -> Vec<Network> {
    let mut networks = Vec::new();

    (0..n).for_each(|i| {
        networks.push(Network::new(format!("net_{}", i) , n_inp, n_hidden, out_dim, activation, out_activation));
    });

    networks
}

fn fitness(net: Network, input: Vec<f64>, target: Vec<f64>) -> f64 {
    let net_out = net.layer_output(input);
    let sum: f64 = target.to_owned().into_iter().enumerate().map(|(i, y)| {
        (y - net_out[i]).powf(2.0)
    }).sum();
    sum / target.len() as f64
}

fn selection(generation: Vec<Network>, input: Vec<f64>, target: Vec<f64>, delta: f64) -> Vec<Network> {
    let fintess_mean = (generation.to_owned().into_iter().map(|net| {
        fitness(net, input.to_owned(), target.to_owned())
    }).sum::<f64>()) / generation.len() as f64;

    let new_generation = generation.to_owned().into_iter().filter(|net| {
        return fitness(net.to_owned(), input.to_owned(), target.to_owned()) - fintess_mean >= delta
    }).collect();

    new_generation
}

fn crossing_over(generation: Vec<Network>) -> Vec<Network> {
    let mut new_generation = Vec::new();

    (0..generation.len() / 2).into_iter().for_each(|_| {
        let rpid1 = rand::thread_rng().gen_range(0..generation.len());
        let rpid2 = rand::thread_rng().gen_range(0..generation.len());

        let parent1 = generation.get(rpid1).unwrap();
        let parent2 = generation.get(rpid2).unwrap();

        let random_split = rand::thread_rng().gen_range(0..parent1.to_owned().weights.len() - 1);
        
        let child1_genes_part_1 = parent1.weights.to_owned().as_slice()[0..random_split].to_vec();
        let child1_genes_part_2 = parent2.weights.to_owned().as_slice()[random_split..parent2.weights.len()].to_vec();
        let child1_ws = [&child1_genes_part_1[..], &child1_genes_part_2[..],].concat();
        
        let child1 = Network::from_flatten(
            child1_ws, 
            rand_name(), 
            parent1.in_weights.len(), 
            parent1.n_hidden, 
            parent1.out_weights.len() / parent1.n_hidden, 
            parent1.activation,
            parent1.out_activation
        );

        let child2_genes_part_1 = parent2.weights.to_owned().as_slice()[0..random_split].to_vec();
        let child2_genes_part_2 = parent1.weights.to_owned().as_slice()[random_split..parent1.weights.len()].to_vec();
        let child2_ws = [&child2_genes_part_1[..], &child2_genes_part_2[..],].concat();

        let child2 = Network::from_flatten(
            child2_ws, 
            rand_name(),
            parent1.in_weights.len(), 
            parent1.n_hidden, 
            parent1.out_weights.len() / parent1.n_hidden, 
            parent1.activation,
            parent1.out_activation
        );

        new_generation.push(child1);
        new_generation.push(child2);
    });

    new_generation
}

fn mutation(generation: Vec<Network>, p: f64) -> Vec<Network> {
    let mut new_generation = Vec::new();

    generation.to_owned().into_iter().for_each(|net| {
        let mut new_net = net.to_owned();
        if rand::thread_rng().gen_range(0..=10) as f64 / 10.0 <= p {
            let mut new_ws = net.weights.to_owned();
            let rw_id = rand::thread_rng().gen_range(0..net.weights.len());
            
            new_ws[rw_id] = weights::Weight {
              value: new_ws[rw_id].value + rand::thread_rng().gen_range(-100000..=100000) as f64 / 100000.0,
              name: new_ws[rw_id].name.to_owned()
            };
          new_net.weights = new_ws;
        } 

        new_generation.push(new_net);
    });

    new_generation
}

fn choise_best(generation: Vec<Network>, inps: Vec<f64>, target: Vec<f64>) -> Network {
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


fn main() -> Result<(), Box<dyn std::error::Error>>  {
    let mut generation = create_generation(5000, 4, 10, 1, ActivationType::ReLU, ActivationType::ReLU);

    let main_input = (0..4).map(|_| { rand::thread_rng().gen_range(0..=1000) as f64 / 1000.0 } ).collect::<Vec<f64>>();

    let target_ws: Vec<weights::Weight> = (0..generation[0].weights.len()).map(|_| weights::Weight::random_weight("name".to_string())).collect();
    
    let target_value = rand::thread_rng().gen_range(0..=1000) as f64 / 1000.0 + rand::thread_rng().gen_range(0..=1000) as f64 / 1000.0;

    println!("targer ws: {:?}", target_ws.to_owned().iter().map(|w| w.value).collect::<Vec<f64>>());
    println!("\ntarget result: {}", target_value);
    let mut best_net = generation[0].to_owned();

    //// MAIN LOOP
    let mut i = 2;
    let mut first_len = generation.to_owned().len();

    while i <= generation.to_owned().len() / 2 {
      let mut bests = Vec::new();
      
      while bests.len() != first_len / i {
        generation = selection(generation.to_owned(), main_input.to_owned(), vec![target_value], 0.01);
        generation = crossing_over(generation.to_owned());
        generation = mutation(generation.to_owned(), 0.7);
        
        let fintess_mean = (generation.to_owned().into_iter().map(|net| {
            fitness(net, main_input.to_owned(), vec![target_value].to_owned())
        }).sum::<f64>()) / generation.len() as f64;
        if fintess_mean.is_nan() {
            let r_generation = create_generation(10, 4, 10, 1, ActivationType::ReLU, ActivationType::ReLU);
            generation.extend(r_generation);
        }

        let best_net_loc = choise_best(generation.to_owned(), main_input.to_owned(), vec![target_value]);
        bests.push(best_net_loc.to_owned());

        generation = generation.into_iter().filter(|n| n.name != best_net_loc.name).collect();
      }

      generation = bests.to_owned();

      first_len = bests.to_owned().len();      
    
      let best_net_loc = choise_best(generation.to_owned(), main_input.to_owned(), vec![target_value]);
      println!("{:?}", best_net_loc.layer_output(main_input.to_owned()));
    
      if (best_net_loc.layer_output(main_input.to_owned())[0] - target_value).abs() < 0.01 {
          println!("{} - break", (best_net_loc.layer_output(main_input.to_owned())[0] - target_value).abs());
          best_net = best_net_loc;
          break;
      }
      if (best_net_loc.layer_output(main_input.to_owned())[0] - target_value).abs() < 0.1 {
        println!("{}", (best_net_loc.layer_output(main_input.to_owned())[0] - target_value).abs());
        i += 2;
      }
    }
    println!("end: {}", i);

    println!("best net output: {:?}", best_net.layer_output(main_input.to_owned()));
    println!("expected output: {:?}", target_value);

    Ok(())
}
