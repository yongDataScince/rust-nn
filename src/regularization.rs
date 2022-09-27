use crate::network::Network;

pub fn l1_regularization(nn: &Network, reg_par: f32) -> f32 {
  (reg_par / nn.weights.len() as f32) * nn.weights.to_owned().into_iter().map(|w| w.value.abs()).sum::<f32>()
}