
use crate::weights::Weight;

#[derive(Debug)]
pub enum LayerType {
  Dense(u32, u32),
  Conv2d(u32, u32, u32, u32), // Input Dim, Output Dim, Window Size, Step Size
  Dropdown(f32)
}

#[derive(Debug)]
pub struct Dense {
  pub inp_dim: u32,
  pub out_dim: u32,
  pub weights: Vec<Weight>
}

impl Dense {
  pub fn new(inp_dim: u32, out_dim: u32) -> Dense {
    Dense {
      inp_dim,
      out_dim
    }
  }
}
