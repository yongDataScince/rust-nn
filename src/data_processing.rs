use std::collections::{HashMap, HashSet};
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Col {
  pub name: String,
  pub values: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Row {
  pub values: HashMap<String, String>,
}

#[derive(Clone)]
pub struct Series  {
  pub headers: Vec<String>,
  pub cols: HashMap<String, Col>,
  pub rows: HashMap<u64, Row>,
}

impl Series {
  pub fn from_csv(path: String, shuffle: bool) ->  Result<Series, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(path)?;

    let mut cols: HashMap<String, Col> = HashMap::new();

    let mut records: Vec<Vec<String>> = reader.deserialize().par_bridge().into_par_iter().map(|res| {
      let rec: Vec<String> = res.expect("error: rec parse");
      rec
    }).collect();
    
    if shuffle {
      records.shuffle(&mut rand::thread_rng());
    }

    let headers = reader.headers()?;

    println!("{:?}", headers);
    headers.to_owned().into_iter().enumerate().for_each(|(id, header)| {
      let mut col = Col { name: header.to_string(), values: vec![] };
      records.to_owned().into_iter().for_each(|rev| {
        col.values.push(rev[id].to_owned());
      });
      cols.insert(header.to_string(), col);
    });

    let mut rows: HashMap<u64, Row> = HashMap::new();

    records.to_owned().into_iter().enumerate().for_each(|(id, values)| {
      let mut row = Row {
        values: HashMap::new()
      };

      values.into_iter().enumerate().for_each(|(i, val)| {
        row.values.insert(headers[i].to_string(), val);
      });

      rows.insert(id as u64, row);
    });

    Ok(Series {
      headers: headers.to_owned().into_iter().map(|h| h.to_string()).collect(),
      cols,
      rows
    })
  }

  pub fn scale_by(&mut self, col_name: &str, value: f32) {
    let curr_col = self.cols.get(col_name).expect("Column does not exist");

    let new_vals: Vec<String> = curr_col.values.to_owned().into_iter().enumerate().map(|(id, v)| {
      let parsed_res = v.parse::<f32>();
      if let Ok(par_val) = parsed_res {
        
        self.rows.entry(id as u64).and_modify(|r| {
          r.values.insert(col_name.to_string(), (par_val / value).to_string());
        });

        return (par_val / value).to_string();
      } else {
        panic!("Cant parse values in current column: {}", v)
      }
    }).collect();
    
    let new_col = Col {
      values: new_vals.to_owned(),
      name: col_name.to_owned()
    };

    self.cols.insert(col_name.to_owned(), new_col);
  }

  pub fn scale_by_max(&mut self, col_name: &str) {
    let curr_col = self.cols.get(col_name).expect("Column does not exist");
    
    let max_value = curr_col.values.to_owned().iter().map(|v| {
      let parsed_res = v.parse::<f32>();
      if let Ok(par_val) = parsed_res {
        par_val
      } else {
        panic!("Cant parse values in current column: {}", v)
      }
    }).max_by(|a, b| a.total_cmp(b)).unwrap();

    (0..self.rows.len()).for_each(|id| {
      self.rows.entry(id as u64).and_modify(|row| {
        row.values.entry(col_name.to_owned()).and_modify(|col_val| {
          let parsed_res = col_val.to_owned().parse::<f32>().unwrap();

          *col_val = (parsed_res / max_value).to_string();
        });
      });
    });

    let new_vals: Vec<String> = curr_col.values.to_owned().into_par_iter().map(|v| {
      let parsed_res = v.parse::<f32>().unwrap();
      return (parsed_res / max_value).to_string();
    }).collect();
  
    self.cols.entry(col_name.to_owned()).and_modify(|col| {
      col.values = new_vals;
    });
  }

  pub fn replace_with(&mut self, col_name: &str, old_vals: Vec<&str>, new_vals: Vec<&str>) {
    let col = self.cols.get_mut(col_name).unwrap();

    old_vals.to_owned().into_iter().enumerate().for_each(|(i, old)| {
      col.values = col.values.to_owned().into_par_iter().map(|v| {
        if v == old {
          return new_vals[i].to_string()
        }
        return v.to_string();
      }).collect();
    });
  }

  pub fn drop_col(&mut self, col_name: &str) -> Vec<String> {
    let droped_col = &self.cols.get(col_name).unwrap().values.to_owned();
    self.headers = self.headers.to_owned().into_iter().filter(|col| *col != col_name).collect();
    self.cols.remove(col_name);
    self.rows = (&self.rows).into_iter().map(|(id, row)| {
      
      let new_row = Row {
        values: row.clone().values.iter().filter(|(col, _)| col_name != *col).map(|(s1, s2)| (s1.to_string(), s2.to_string())).collect()
      };

      return (
        *id,
        new_row
      );
    }).collect();

    droped_col.to_owned()
  }

  pub fn to_vecs(&self) -> Vec<Vec<f32>> {
    (0..self.rows.len()).map(|id| {
      let row = &self.rows.get(&(id as u64)).unwrap();
      let values: Vec<f32> = self.headers.iter().map(|header| row.values.get(header).unwrap().parse::<f32>().expect("Error to parse value")).collect();

      values
    }).collect()
  }

  pub fn batchise(data: Vec<Vec<f32>>, batch_size: usize) -> Vec<Vec<Vec<f32>>> {
    data.chunks(batch_size).map(|chunk| chunk.to_vec()).collect()
  }

  pub fn max_by(&self, col_name: &str) -> f32 {
    self.cols.get(col_name).unwrap().values.iter().map(|val| val.parse::<f32>().unwrap()).max_by(|a, b| a.total_cmp(b)).unwrap()
  }

  pub fn min_by(&self, col_name: &str) -> f32 {
    self.cols.get(col_name).unwrap().values.iter().map(|val| val.parse::<f32>().unwrap()).min_by(|a, b| a.total_cmp(b)).unwrap()
  }

  pub fn mean_by(&self, col_name: &str) -> f32 {
    self.cols.get(col_name).unwrap().values.iter().map(|val| val.parse::<f32>().unwrap()).sum::<f32>() / self.cols.get(col_name).unwrap().values.len() as f32
  }

  pub fn std_by(&self, col_name: &str) -> f32 {
    let mean = self.mean_by(col_name);

    let sum: f32 = self.cols.get(col_name).unwrap().values.iter().map(|val| {
      let x = val.parse::<f32>().unwrap();
      
      (x - mean).powf(2.0)
    }).sum();

    (sum / self.cols.get(col_name).unwrap().values.len() as f32).sqrt()
  }

  pub fn sub_by(&mut self, col_name: &str, value: f32) {
    let curr_col = self.cols.get(col_name).expect("Column does not exist");

    let new_vals: Vec<String> = curr_col.values.to_owned().into_iter().enumerate().map(|(id, v)| {
      let parsed_res = v.parse::<f32>();
      if let Ok(par_val) = parsed_res {
        
        self.rows.entry(id as u64).and_modify(|r| {
          r.values.insert(col_name.to_string(), (par_val - value).to_string());
        });

        return (par_val - value).to_string();
      } else {
        panic!("Cant parse values in current column: {}", v)
      }
    }).collect();
    
    let new_col = Col {
      values: new_vals.to_owned(),
      name: col_name.to_owned()
    };

    self.cols.insert(col_name.to_owned(), new_col);
  }

  pub fn draw_col(&self, col_name: &str) {
    let img_name = &format!("graphs/{col_name}.png");

    let vals: Vec<f32> = self.cols.get(col_name).unwrap().values.to_owned().into_iter().map(|val| val.parse().unwrap()).collect();
    
    let root_area = BitMapBackend::new(img_name, (1200, 600)).into_drawing_area();
    
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
      .set_label_area_size(LabelAreaPosition::Left, 40)
      .set_label_area_size(LabelAreaPosition::Bottom, 40)
      .caption("Loss", ("sans-serif", 40))
      .build_cartesian_2d(-0..(vals.len() + 20) as i32, -2.0..10.0f32)
      .unwrap();
      
      ctx.configure_mesh().draw().unwrap();
      
      let series_err = LineSeries::new(
        vals.iter().enumerate().map(|(i, v)| {
          return (i as i32, *v);
        }).collect::<Vec<(i32, f32)>>(),
        &RED
      );

      ctx.draw_series(series_err).unwrap();
  }

  pub fn unique_in_col(&self, col_name: &str) -> Vec<String> {
    self.cols.get(col_name).unwrap().values.iter()
    .map(|v| v.to_string())
    .collect::<HashSet<String>>()
    .into_iter()
    .collect()
  }
}