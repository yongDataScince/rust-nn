pub mod loss;
pub mod activation;
pub mod errors;
pub mod layer;
pub mod weights;
pub mod network;
pub mod loader;

use crate::{
    layer::Layer,
    activation::ActivationType,
    loader::read_from_file,
};

fn main() -> Result<(), Box<dyn std::error::Error>>  {

    let records = read_from_file("Iris.csv")?; // classification

    let values = records.iter().map(|r| r.as_slice()[1..r.len() - 1].to_vec()).collect::<Vec<Vec<f64>>>();
    let answers = records.iter().map(|r| vec![r[r.len() - 1]] ).collect::<Vec<Vec<f64>>>();

    let mut layer1 = Layer::new(values[0].len(), 7, 1, ActivationType::Sigmoid, ActivationType::Sigmoid);

    println!("n_inp: {}", values[0].len());
    println!("n_out: {}", answers[0].len());

    let train_len = ((values.len() as f64 / 100.0) * 85.0).round() as usize;
    let test_len = ((values.len()as f64 / 100.0) * 15.0) as usize;

    println!("train len: {}\ntest len: {}", train_len, test_len);

    let train_data = values.to_owned().as_slice()[0..train_len].to_vec();
    let train_answers = answers.to_owned().as_slice()[0..train_len].to_vec();

    let test_data = values.to_owned().as_slice()[0..test_len].to_vec();
    let test_answers = answers.to_owned().as_slice()[0..test_len].to_vec();

    for e in 0..1000 {

        let out = layer1.train_layer(
            values.to_owned(),
            &answers,
            0.0001
        );
        println!("Thead #1 err: {}", out.error);
    }

    let mut right_count = 0;
    for i in 0..test_len {
        let out = ActivationType::step(layer1.layer_output(test_data[i].to_owned())[0]);
        if test_answers[i][0] == out {
            right_count += 1;
        }
    }
    
    println!("accuracy no batches: {}/{}", right_count, test_len);
    println!("\n");
    Ok(())
}
