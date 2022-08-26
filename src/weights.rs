use rand::*;

#[derive(Debug, Clone, Copy)]
pub struct Weight {
  pub value: f64
}

impl Weight {
    pub fn random_weight() -> Weight {
      Weight { value: thread_rng().gen_range(-10000..=10000) as f64 / 10000.0 }
    }
}
