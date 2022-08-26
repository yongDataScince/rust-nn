pub mod neuron;
pub mod activation;
pub mod errors;
pub mod layer;
pub mod weights;

use crate::{
    layer::Layer,
    activation::ActivationType,
};

fn main() {
    let layer1 = Layer::new(2, 3, ActivationType::Sigmoid);
    let layer2 = Layer::new(3, 1, ActivationType::Sigmoid);
    println!("{:?}", layer1.neurons);
    println!("{:?}", layer2.neurons);
}
