use rand::Rng;

pub fn argmax<T: Copy + PartialOrd>(u: &[T]) -> usize {
  assert!(u.len() != 0);

  let mut max_index = 0;
  let mut max = u[max_index];

  for (i, v) in (u.iter()).enumerate() {
      if max < *v {
          max_index = i;
          max = *v;
      }
  }

  max_index
}

pub fn rand_name() -> String {
  rand::thread_rng()
      .sample_iter(&rand::distributions::Alphanumeric)
      .take(7)
      .map(char::from)
      .collect()
}
