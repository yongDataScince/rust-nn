pub fn read_from_file(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
  let mut reader = csv::Reader::from_path(path)?;

  let mut records = Vec::new();

  let headers = reader.headers()?;
  println!("{:?}", headers);

  for res in reader.deserialize() {
    let rec: Vec<f64> = res?;
    records.push(rec)
  }

  Ok(records)
}