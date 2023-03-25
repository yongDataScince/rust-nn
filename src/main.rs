pub mod preludes;
pub mod loss;
pub mod activation;
pub mod errors;
pub mod network;
pub mod weights;
pub mod loader;
pub mod data_processing;
pub mod enviroment;


use activation::ActivationType;
use ndarray::Shape;
use ndarray::prelude::*;
use network::Network;
use rand::Rng;

use crate::data_processing::Series;
use crate::loss::*;
use crate::preludes::*;

extern crate piston;
extern crate graphics;
extern crate glutin_window;
extern crate opengl_graphics;

////-- CONSTANTS --/////

const POP_SIZE: usize = 20;
const MAX_GEN: u32 = 100;
const P_CROSSOW: f64 = 0.5;
const P_MUTATION: f64 = 0.15;

////////////////////////

fn all_unique<T>(vals: &Vec<T>) -> bool where T: PartialEq + PartialOrd + Clone {
    let mut prev_value = vals[0].clone();
    let mut uniq = true;
    for i in 1..vals.len() {
        for j in 1..vals.len() {
            uniq = prev_value == vals[j];
        }
        prev_value = vals[i].clone();
    }

    uniq
}

fn toutnament(vals: Vec<f64>, n_leaders: usize, p_len: usize) -> Vec<usize> {
    let mut out_ids = Vec::new();
    for _ in 0..p_len {
        let mut ids = vec![0; n_leaders];

        let mut i = 0;
        while all_unique::<usize>(&ids)  {
            ids[i] = rand::thread_rng().gen_range(0..vals.len());

            if i == n_leaders - 1 {
                i = 0;
            } else {
                i += 1;
            }
        }
        let best = ids.clone().into_iter().max_by(|id1, id2| {
            vals[id1.clone()].partial_cmp(&vals[id2.clone()]).unwrap()
        }).unwrap();

        out_ids.push(best);
    }

    out_ids
}

fn mutate_weigths(ws: &mut Vec<(String, String, f64)>) {
    *ws = ws.clone().into_iter().map(|w| {
        let p = rand::thread_rng().gen_range(0.0..=100.0) / 100.0;
        let mut w = w.clone();
        if p <= P_MUTATION {
            let v = rand::thread_rng().gen_range(-50.0..=200.0) / 100.0;
            w.2 += v;
        }
        w
    }).collect::<Vec<(String, String, f64)>>();
}

fn crossover(par1: &Network, par2: &Network) -> (Network, Network) {

    let mut ch1 = par1.clone();
    let mut ch2 = par2.clone();

    let par1_ws = par1.weights_to_vec();
    let par2_ws = par2.weights_to_vec();

    let rand_id = rand::thread_rng().gen_range(2..par1_ws.len() - 3);

    let mut gen1 = par1_ws[0..rand_id].to_vec().clone();
    let gen2 = par2_ws[rand_id..par1_ws.len()].to_vec().clone();
    gen1.extend(gen2);
    ch1.import_ws(gen1);

    
    let mut gen2 = par2_ws[0..rand_id].to_vec().clone();
    let gen1 = par1_ws[rand_id..par1_ws.len()].to_vec().clone();
    gen2.extend(gen1);
    ch2.import_ws(gen2);

    (ch1, ch2)
}

fn main() -> Result<(), Box<dyn std::error::Error>>  {
    Ok(())
}

/*
def convolve_1d(array, kernel):
    ks = kernel.shape[0] # shape gives the dimensions of an array, as a tuple
    final_length = array.shape[0] - ks + 1
    return numpy.array([(array[i:i+ks]*kernel).sum() for i in range(final_length)])
*/
