pub fn read_from_file(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
  let mut reader = csv::Reader::from_path(path)?;

  let mut records = Vec::new();

  for res in reader.deserialize() {
    let rec: Vec<f64> = res?;
    let max = rec.to_owned().into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let content = rec.to_owned().into_iter().enumerate().map(|(id, v)| { 
      if id != 0 {
        return v / max;
      } else {
        v
      }
    }).collect();
    records.push(content)
  }

  Ok(records)
}