use rand::*;

#[derive(Debug, Clone)]
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
      Weight { name, value: thread_rng().gen_range(-10000..=10000) as f64 / 10000.0 }
    }
}

impl std::fmt::Display for Weight {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "name: {}", self.name)
  }
}
