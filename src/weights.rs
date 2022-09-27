use rand::*;
use serde::{ Deserialize, Serialize };

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Weight {
  pub value: f32,
  pub name: String
}

impl PartialEq for Weight  {
    fn eq(&self, other: &Self) -> bool {
      self.name == other.name
    }
}

impl Weight {
    pub fn random_weight(name: String) -> Weight {
      Weight { name, value: thread_rng().gen_range(-100000..=100000) as f32 / 100000.0 }
    }
}

impl std::fmt::Display for Weight {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "name: {}", self.name)
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Bias {
  pub value: f32,
  pub name: String
}

impl PartialEq for Bias  {
    fn eq(&self, other: &Self) -> bool {
      self.name == other.name
    }
}

impl Bias {
    pub fn random_bias(name: String) -> Bias {
      Bias { name, value: thread_rng().gen_range(1000..=1000) as f32 / 10000.0 }
    }
}

impl std::fmt::Display for Bias {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "name: {}", self.name)
  }
}
