use rand::*;
use serde::{ Deserialize, Serialize };

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weight {
  pub value: f64,
  pub name: String,
}

impl PartialEq for Weight  {
    fn eq(&self, other: &Self) -> bool {
      self.name == other.name
    }
}

impl Weight {
    pub fn random_weight(name: String) -> Weight {
      let xavier_noise_lower = -(1.0 / 10.0_f64.powf(1.0 / 10.0));
      let xavier_noise_apper = 1.0 / 10.0_f64.powf(1.0 / 10.0);

      let rand = xavier_noise_lower + (thread_rng().gen_range(0..=11000) as f64 / 10000.0) * (xavier_noise_apper - xavier_noise_lower);

      Weight { name, value: rand }
    }
}

impl std::fmt::Display for Weight {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "name: {}", self.name)
  }
}
