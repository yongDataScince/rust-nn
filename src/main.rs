pub mod loss;
pub mod activation;
pub mod errors;
pub mod layer;
pub mod weights;
pub mod network;
pub mod loader;
use plotters::prelude::*;
use rand::Rng;

use crate::{
    layer::Layer,
    activation::ActivationType,
    loader::read_from_file,
};

fn num_to_onehot(num: u32, max_num: u32) -> Vec<f64> {
    let mut zeros: Vec<f64> = vec![0.0;max_num as usize];

    zeros[num as usize] = 1.0;

    zeros
}

fn argmax<T: Copy + PartialOrd>(u: &[T]) -> (usize, T) {
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

fn main() -> Result<(), Box<dyn std::error::Error>>  {
    let batch_suze = 16;

    let records = read_from_file("test.csv")?;

    let values = records.iter().map(|r| r.as_slice()[0..r.len() - 1].to_vec()).collect::<Vec<Vec<f64>>>();
    let answers = records.iter().map(|r| vec![r[r.len() - 1]] ).collect::<Vec<Vec<f64>>>();

    let one_hot_answers = answers.to_owned().into_iter().map(|n| num_to_onehot(n[0] as u32, 10)).collect::<Vec<Vec<f64>>>();

    let mut layer1 = Layer::new(values[0].len(), 7, 10, ActivationType::ReLU, ActivationType::Softmax);

    // layer1.load_model("layers/700_0.json".to_string());

    println!("n_inp: {}", values[0].len());
    println!("n_out: {}", answers[0].len());

    let train_len = ((values.len() as f64 / 100.0) * 85.0).round() as usize;
    let test_len = ((values.len()as f64 / 100.0) * 15.0) as usize;

    println!("train len: {}\ntest len: {}", train_len, test_len);

    let train_data = values.to_owned().as_slice()[0..train_len].to_vec();
    let train_answers = one_hot_answers.to_owned().as_slice()[0..train_len].to_vec();

    println!("{:?}", train_data[0]);
    println!("{:?}", train_answers[0]);

    let o = layer1.layer_output(train_data[0].to_owned());
    println!("{o:?}");

    let test_data = values.to_owned().as_slice()[0..test_len].to_vec();
    let test_answers = one_hot_answers.to_owned().as_slice()[0..test_len].to_vec();

    let train_data_chunks = train_data.to_owned().chunks(batch_suze).map(|chunk| chunk.to_vec()) .collect::<Vec<Vec<Vec<f64>>>>();
    let train_answers_chunks = train_answers.to_owned().chunks(batch_suze).map(|chunk| chunk.to_vec()) .collect::<Vec<Vec<Vec<f64>>>>();

    let mut errors: Vec<f64> = Vec::new();
    for e in 0..10 {
        train_data_chunks.to_owned().into_iter().enumerate().for_each(|(i, chunck)| {
            let out = layer1.train_layer(
                chunck.to_owned(),
                &train_answers_chunks[i],
                0.0003
            );
            if i % 2 == 0 {
                errors.push(out.error)
            }
            if i % 100 == 0 {
                (0..10).for_each(|i| {
                    let ri = rand::thread_rng().gen_range(0..test_answers.len());

                    let t_o = layer1.layer_output(test_data[ri].to_owned());
                    let t_arg = argmax(t_o.as_slice());
                    println!("test output: {t_arg:?}");
                    println!("true output: {}", test_answers[ri][0]);
                });

                layer1.save_weights(format!("layers/{i}_{e}.json"));
                let img_name = &format!("images/{i}_{e}.png");
                
                let root_area = BitMapBackend::new(img_name, (1200, 600))
                    .into_drawing_area();
                root_area.fill(&WHITE).unwrap();

                let mut ctx = ChartBuilder::on(&root_area)
                    .set_label_area_size(LabelAreaPosition::Left, 40)
                    .set_label_area_size(LabelAreaPosition::Bottom, 40)
                    .caption("Loss", ("sans-serif", 40))
                    .build_cartesian_2d(-0..(errors.len() + 20) as i32, -2.0..10.0)
                    .unwrap();
                
                ctx.configure_mesh().draw().unwrap();
                let series = LineSeries::new(
                    errors.iter().enumerate().map(|(i, v)| {
                        return (i as i32, *v);
                    }).collect::<Vec<(i32, f64)>>(),
                    &GREEN
                );
                
                ctx.draw_series(series).unwrap();
            }
            println!("{i}. err: {}", out.error);
        });
    }

    let mut right_count = 0;
    for i in 0..test_len {
        let out = ActivationType::step(layer1.layer_output(test_data[i].to_owned())[0]);
        if test_answers[i][0] == out {
            right_count += 1;
        }
    }
    
    println!("accuracy no batches: {}/{}", right_count, test_len);
    println!("\n");
    Ok(())
}
