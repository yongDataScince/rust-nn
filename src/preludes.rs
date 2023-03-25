use ndarray::{Array, Dim};

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

pub fn vec_to_array<T: Clone>(v: Vec<Vec<T>>) -> Array<T, Dim<[usize; 2]>> {
  if v.is_empty() {
      return Array::from_shape_vec((0, 0), Vec::new()).unwrap();
  }
  let nrows = v.len();
  let ncols = v[0].len();
  let mut data = Vec::with_capacity(nrows * ncols);
  for row in &v {
      assert_eq!(row.len(), ncols);
      data.extend_from_slice(&row);
  }
  Array::from_shape_vec((nrows, ncols), data).unwrap()
}