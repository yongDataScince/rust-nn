use rand::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Weight {
  pub value: f64,
  pub name: String,
}

impl Weight {
    pub fn random_weight(name: String) -> Weight {
      Weight { name, value: thread_rng().gen_range(-10000..=10000) as f64 / 10000.0 }
    }
}
