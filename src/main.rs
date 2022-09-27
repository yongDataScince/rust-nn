pub mod loss;
pub mod activation;
pub mod errors;
pub mod network;
pub mod weights;
pub mod loader;
pub mod data_processing;
pub mod regularization;

use std::time::Duration;

use plotters::prelude::*;
use crate::{
    loss::{
        cross_entropy_loss, loss_mse, binary_cross_entropy_loss
    },
    data_processing::Series
};
use crate::{
    network::Network,
    activation::ActivationType,
};

// series.headers.to_owned().into_iter().for_each(|head| {
    //     let mean_data = series.mean_by(&head);
    //     let std = series.std_by(&head);

    //     series.sub_by(&head, mean_data);
    //     series.scale_by(&head, std);
    // });
    // println!("{}", series.max_by("Area"));

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


fn num_to_onehot(num: u32, max_num: u32) -> Vec<f32> {
    let mut zeros: Vec<f32> = vec![0.0;max_num as usize];

    zeros[num as usize] = 1.0;

    zeros
}

fn main() -> Result<(), Box<dyn std::error::Error>>  {
    let batch_size = 8;
    let mut series: Series = Series::from_csv("data/Dry_Bean_Dataset.csv".to_string(), true)?;    
    println!("{:?}", series.unique_in_col("Class"));
    series.draw_col("Bean ID");
    series.replace_with("Class",
      vec!["HOROZ", "CALI", "BARBUNYA", "SIRA", "DERMASON", "SEKER", "BOMBAY"],
      vec!["0", "1", "2", "3", "4", "5", "6"]
    );

    let answers = series.drop_col("Class")
        .into_iter()
        .map(|v| {
            num_to_onehot(v.parse::<u32>().unwrap(), 7)
        }).collect::<Vec<Vec<f32>>>();
    let answers = Series::batchise(answers, batch_size);

    series.headers.to_owned().into_iter().for_each(|head| {
        let min_value = series.min_by(&head);
        let max_value = series.max_by(&head);

        series.sub_by(&head, min_value);
        series.scale_by(&head, max_value - min_value);
    });

    let vec_data = series.to_vecs();
    let batch_vectorized_data = Series::batchise(vec_data, batch_size);
    
    ///////// TEST
    // let mut test_series: Series = Series::from_csv("data/Dry_Bean_Dataset.csv".to_string(), true)?;    

    // let test_answers = test_series.drop_col("species")
    //     .into_iter()
    //     .map(|v| {
    //         num_to_onehot(v.parse::<u32>().unwrap(), 3)
    //     }).collect::<Vec<Vec<f32>>>();

    // test_series.headers.to_owned().into_iter().for_each(|head| {
    //     let min_value = test_series.min_by(&head);
    //     let max_value = test_series.max_by(&head);

    //     test_series.sub_by(&head, min_value);
    //     test_series.scale_by(&head, max_value - min_value);
    // });

    // let vec_data_test = test_series.to_vecs();
    /////////
    
    let mut net = Network::new(vec![
        ("Input".to_string(), 16, 20, ActivationType::ReLU, 0.0),
        ("Hid1".to_string(), 20, 10, ActivationType::ReLU, 0.0),
        ("Out".to_string(), 10, 7, ActivationType::Softmax, 0.0)
    ]);
    println!("weigths len: {}", net.weights.len());

    for e in 0..40 {
        
        for i in 0..batch_vectorized_data.len() {
            let o = net.train_layer(
                &cross_entropy_loss,
                batch_vectorized_data[i].to_owned(),
                &answers[i],
                0.001,
                1.0,
                false
            );
            if i % 8 == 0 {
                let out = net.output(&batch_vectorized_data[i][0].to_owned(), false);
                println!("{:.4?} - {:?}", out, &answers[i][0]);
            }
            println!("{e}:{i} err: {}", o.error);
        }
    }

    // for i in 0..vec_data_test.len() {
    //     let nn_out = net.output(&vec_data_test[i], false);
    //     println!("{:?} - {:?}", num_to_onehot(
    //         argmax(&nn_out).0 as u32,
    //         3
    //     ), test_answers[i]);
    // }

    // let mut nn_data = Vec::new();
    // (1000..2000).for_each(|i| {
    //     let x_add = rand::Rng::gen_range(&mut rand::thread_rng(), -10000..=100000) as f32 / 1000.0;
    //     let y_add = rand::Rng::gen_range(&mut rand::thread_rng(), -10000..=100000) as f32 / 1000.0;

    //     let out = net.output(&vec![(x_add + i as f32) / 1000.0, (y_add + i as f32) / 1000.0]);

    //     nn_data.push([out[0] * 1000.0, x_add + i as f32, y_add + i as f32]);
    // });

    // let area = BitMapBackend::gif(
    //     "images/animated.gif", 
    //     (640, 480), 
    //     10
    // ).unwrap().into_drawing_area();
    
    // for i in 0..=200 {
    //     area.fill(&WHITE).unwrap();

    //     let mut chart = ChartBuilder::on(&area)
    //         .margin(20)
    //         .caption("Empty 3D Figure", ("sans-serif", 40))
    //         .build_cartesian_3d::<std::ops::Range<f32>, std::ops::Range<f32>, std::ops::Range<f32>>(0.0..2000.0, 0.0..2000.0, 0.0..2000.0)
    //         .unwrap();

    //     chart.with_projection(|mut pb| {
    //         pb.pitch = 0.0;
    //         pb.yaw = i as f32 / 10.0;
    //         pb.scale = 0.5;
    //         pb.into_matrix()
    //     });

    //     let dr_data = all_data.to_owned().into_iter().map(|slice| (slice[0], slice[1], slice[2]));
    //     chart.draw_series(LineSeries::new(
    //         dr_data,
    //         &RED
    //     )).unwrap();

    //     let dr_data2 = nn_data.to_owned().into_iter().map(|slice| (slice[0], slice[1], slice[2]));
    //     chart.draw_series(LineSeries::new(
    //         dr_data2,
    //         &GREEN
    //     )).unwrap();
        
    //     area.present().unwrap();
    // }
    // let out = net.output(&batch_vectorized_data[0][0]);
    // println!("{:?}", out);

    // let mut errors: Vec<f32> = Vec::new();

    // for e in 0..200 {
    //     batch_vectorized_data.to_owned().into_iter().enumerate().for_each(|(i, chunck)| {
    //         let out = net.train_layer(
    //             &loss_mse,
    //             chunck.to_owned(),
    //             &labels[i],
    //             0.0001
    //         );
    //         println!("{i}. err: {}", out.error);
    //         errors.push(out.error);
    //         if i % (batch_size / 2) == 0 {
    //             let a_c: u32 = (0..batch_size).map(|i| {
    //                 let output = net.output(&batch_vectorized_data[0][i].to_owned());
    //                 let arg_output =  argmax(output.as_slice());
    //                 let arg_answer = argmax(labels[0][i].to_owned().as_slice());
    //                 println!("net out: {:?}\nnet out(oh): {:?}\ntrue answ: {:?}\n\n", output, num_to_onehot(argmax(output.as_slice()).0.try_into().unwrap(), 7), labels[0][i]);
    //                 return (arg_output.0 == arg_answer.0) as u32
    //             }).sum();
    //             println!("{a_c} / 200");
    //             let img_name = &format!("images/{i}_{e}.png");
                
    //             let root_area = BitMapBackend::new(img_name, (1200, 600))
    //                 .into_drawing_area();
    //             root_area.fill(&WHITE).unwrap();

    //             let mut ctx = ChartBuilder::on(&root_area)
    //                 .set_label_area_size(LabelAreaPosition::Left, 40)
    //                 .set_label_area_size(LabelAreaPosition::Bottom, 40)
    //                 .caption("Loss", ("sans-serif", 40))
    //                 .build_cartesian_2d(-0..(errors.len() + 20) as i32, -2.0..10.0)
    //                 .unwrap();
                
    //             ctx.configure_mesh().draw().unwrap();
    //             let series_err = LineSeries::new(
    //                 errors.iter().enumerate().map(|(i, v)| {
    //                     return (i as i32, *v);
    //                 }).collect::<Vec<(i32, f32)>>(),
    //                 &RED
    //             );

    //             ctx.draw_series(series_err).unwrap();
    //         }
    //     });
    // }

    Ok(())
}
