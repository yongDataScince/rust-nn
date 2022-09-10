pub mod loss;
pub mod activation;
pub mod errors;
pub mod network;
pub mod weights;
pub mod loader;
pub mod utils;
pub mod stand;
pub mod generative;


use crate::stand::Enviroment;

pub struct GenResult {
    name: String,
    value: Vec<f64>
}

#[macroquad::main("Snake")]
async fn main() -> Result<(), Box<dyn std::error::Error>>  {
    let mut envir = Enviroment::new(32, 20);
    envir.run().await;

    // let main_input = (0..4).map(|_| { rand::thread_rng().gen_range(0..=1000) as f64 / 1000.0 } ).collect::<Vec<f64>>();

    // let target_ws: Vec<weights::Weight> = (0..generation[0].weights.len()).map(|_| weights::Weight::random_weight("name".to_string())).collect();
    
    // let target_value = rand::thread_rng().gen_range(0..=1000) as f64 / 1000.0 + rand::thread_rng().gen_range(0..=1000) as f64 / 1000.0;

    // println!("targer ws: {:?}", target_ws.to_owned().iter().map(|w| w.value).collect::<Vec<f64>>());
    // println!("\ntarget result: {}", target_value);
    // let mut best_net = generation[0].to_owned();

    // //// MAIN LOOP
    // let mut i = 2;
    // let mut first_len = generation.to_owned().len();

    // while i <= generation.to_owned().len() / 2 {
    //   let mut bests = Vec::new();
      
    //   while bests.len() != first_len / i {
    //     generation = selection(generation.to_owned(), main_input.to_owned(), vec![target_value], 0.01);
    //     generation = crossing_over(generation.to_owned());
    //     generation = mutation(generation.to_owned(), 0.7);
        
    //     let fintess_mean = (generation.to_owned().into_iter().map(|net| {
    //         fitness(net, main_input.to_owned(), vec![target_value].to_owned())
    //     }).sum::<f64>()) / generation.len() as f64;
    //     if fintess_mean.is_nan() {
    //         let r_generation = create_generation(10, 4, 10, 1, ActivationType::ReLU, ActivationType::ReLU);
    //         generation.extend(r_generation);
    //     }

    //     let best_net_loc = choise_best(generation.to_owned(), main_input.to_owned(), vec![target_value]);
    //     bests.push(best_net_loc.to_owned());

    //     generation = generation.into_iter().filter(|n| n.name != best_net_loc.name).collect();
    //   }

    //   generation = bests.to_owned();

    //   first_len = bests.to_owned().len();      
    
    //   let best_net_loc = choise_best(generation.to_owned(), main_input.to_owned(), vec![target_value]);
    //   println!("{:?}", best_net_loc.layer_output(main_input.to_owned()));
    
    //   if (best_net_loc.layer_output(main_input.to_owned())[0] - target_value).abs() < 0.01 {
    //       println!("{} - break", (best_net_loc.layer_output(main_input.to_owned())[0] - target_value).abs());
    //       best_net = best_net_loc;
    //       break;
    //   }
    //   if (best_net_loc.layer_output(main_input.to_owned())[0] - target_value).abs() < 0.1 {
    //     println!("{}", (best_net_loc.layer_output(main_input.to_owned())[0] - target_value).abs());
    //     i += 2;
    //   }
    // }
    // println!("end: {}", i);

    // println!("best net output: {:?}", best_net.layer_output(main_input.to_owned()));
    // println!("expected output: {:?}", target_value);

    Ok(())
}
