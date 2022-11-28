pub fn argmax<T: Copy + PartialOrd>(u: &[T]) -> (usize, T) {
  assert!(u.len() != 0);
  let mut max_index = 0;
  let mut max = u[max_index];
  for (i, v) in (u.iter()).enumerate() {
      if max < *v {
          max_index = i;
          max = *v;
      }
  }
  (max_index, max)
}


pub fn num_to_onehot(num: u32, max_num: u32) -> Vec<f64> {
  let mut zeros: Vec<f64> = vec![0.0;max_num as usize];
  zeros[num as usize] = 1.0;
  zeros
}