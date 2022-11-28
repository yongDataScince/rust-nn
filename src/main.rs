pub mod preludes;
pub mod loss;
pub mod activation;
pub mod errors;
pub mod network;
pub mod weights;
pub mod loader;
pub mod data_processing;
pub mod enviroment;

use activation::ActivationType;

use crate::data_processing::Series;
use crate::loss::*;
use crate::preludes::*;

extern crate piston;
extern crate graphics;
extern crate glutin_window;
extern crate opengl_graphics;

fn main() -> Result<(), Box<dyn std::error::Error>>  {
    

    Ok(())
}


/*
let batch_size = 16;

    let mut nn = network::Network::new(vec![
        ("Input", 17, 10, ActivationType::Sigmoid),
        ("Hod1", 10, 8, ActivationType::ReLU),
        ("Output", 8, 7, ActivationType::Softmax),
    ]);

    println!("Weight count: {}", nn.weigth_count());
    
    let mut dataset = data_processing::Series::from_csv("./data/Dry_Bean_Dataset.csv", true).unwrap();
    
    // Drop IDs
    dataset.draw_col("Bean ID");

    // Collect classnames
    let class_names = dataset.unique_in_col("Class").iter().map(|name| name.clone()).collect::<Vec<String>>();
    
    // Collect classnames ids
    let values = dataset.unique_in_col("Class").iter().enumerate().map(|(i, _)| {
        i.to_string()
    }).collect::<Vec<String>>();

    // Replace classnames with ids
    dataset.replace_with("Class", class_names.clone(), values);

    // Ids to Onehot 
    let answers: Vec<Vec<f64>> = dataset.drop_col("Class").iter().map(|val| {
        let num = val.parse().unwrap();
        num_to_onehot(num, class_names.clone().len() as u32)
    }).collect();

    let answers = Series::batchise(answers, batch_size);
    
    dataset.scale_by_max("Area");
    dataset.scale_by_max("Perimeter");
    dataset.scale_by_max("MajorAxisLength");
    dataset.scale_by_max("MinorAxisLength");
    dataset.scale_by_max("ConvexArea");
    dataset.scale_by_max("EquivDiameter");

    let data = dataset.to_vecs();
    let data = Series::batchise(data, batch_size);

    println!("Inp len: {}", data[0][0].len());
    println!("Out len: {}", answers[0][0].len());

    for epoch in 0..10 {
        for batch_i in 0..data.len() {
            let res = nn.train_layer(&binary_cross_entropy_loss, data[batch_i].clone(), &answers[batch_i], 1e-4);
            println!("epoch: {epoch} batch: {batch_i} error: {}", res.error);

            if batch_i % 5 == 0 {
                let inp_data = data[batch_i][0].clone();
                let nn_out = nn.output(&inp_data);
                let (to, _) = argmax(&answers[batch_i][0].clone());
                let (no, _) = argmax(&nn_out);

                println!("true: {to}, nn: {no}\nOut vec: {:?}", nn_out);
            }
        }
    }
*/