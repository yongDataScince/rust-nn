pub fn argmax(inp: Vec<f64>) -> f64 {
  inp.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    .map(|(index, _)| index).unwrap() as f64
}