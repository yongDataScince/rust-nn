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

    let records = read_from_file("Iris.csv")?;

    let mut layer = Layer::new(4, ActivationType::Sigmoid);

    let values = records.iter().map(|r| r.as_slice()[1..r.len() - 1].to_vec()).collect::<Vec<Vec<f64>>>();
    let answers = records.iter().map(|r| r[r.len() - 1] ).collect::<Vec<f64>>();

    for _ in 0..10000 {
        let mut preds = Vec::new();
        for data in &values {
            let out = layer.layer_output(data.to_owned());
            preds.push(out);
        }

        layer.train_layer(
            values.to_owned(),
            &answers,
            0.001
        );
    }
    for data in &values {
        let out = layer.layer_output(data.to_owned());
        println!("out of: {:?} = {}", data, out)
    }
    Ok(())
}
