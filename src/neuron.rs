use rand::*;
use crate::{
  activation::ActivationType,
  errors::NNErrors
};

#[derive(Debug, Clone, Copy)]
pub struct Neuron {
  pub value: f64,
  pub activation: ActivationType
}

impl std::fmt::Display for Neuron {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.value)
  }
}

impl std::ops::Add for Neuron {
    type Output = Result<Neuron, NNErrors>;

    fn add(self, rhs: Self) -> Self::Output {
      return Ok(Neuron {
        activation: self.activation,
        value: self.value + rhs.value
      })
    }
}

impl std::ops::Sub for Neuron {
  type Output = Result<Neuron, NNErrors>;

  fn sub(self, rhs: Self) -> Self::Output {
    return Ok(
      Neuron {
        activation: self.activation,
        value: self.value - rhs.value
      }
    )
  }
}

impl std::ops::Mul for Neuron {
  type Output = Result<Neuron, NNErrors>;

  fn mul(self, rhs: Self) -> Self::Output {
    return Ok(Neuron {
      activation: self.activation,
      value: self.value * rhs.value
    })
  }
}

impl std::ops::Div for Neuron {
  type Output = Result<Neuron, NNErrors>;

  fn div(self, rhs: Self) -> Self::Output {
    return Ok(Neuron {
      activation: self.activation,
      value: self.value / rhs.value
    })
  }
}

impl Neuron {
  pub fn new(value: f64, activation: ActivationType) -> Neuron {
    use ActivationType::*;

    match activation {
      Step => Neuron {
        value: if value > 0.5 { 1.0 } else { 0.0 },
        activation
      },
      Sigmoid => Neuron {
        value: ActivationType::sigmoid(value),
        activation
      },
      Tanh => Neuron {
        value: ActivationType::tanh(value),
        activation
      }
    }
  }
  pub fn rand_neuron(activation: ActivationType) -> Neuron {
    use ActivationType::*;

    let value = thread_rng().gen_range(-10000..=10000) as f64 / 10000.0;
    match activation {
      Step => Neuron {
        value: if value > 0.5 { 1.0 } else { 0.0 },
        activation
      },
      Sigmoid => Neuron {
        value: ActivationType::sigmoid(value),
        activation
      },
      Tanh => Neuron {
        value: ActivationType::tanh(value),
        activation
      }
    }
  }

  pub fn zero_neuron(activation: ActivationType) -> Neuron {  
    Neuron {
      value: 0.0,
      activation
    }
  }
}
