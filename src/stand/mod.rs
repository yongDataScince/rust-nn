use crate::generative::{crossing_over, mutation};
use crate::{
    network::Network,
    activation::ActivationType,
};
use crate::utils::argmax;
use macroquad::prelude::*;
use std::collections::LinkedList;

type Point = (i16, i16);

fn create_generation(
  n: usize,
  n_inp: usize,
  n_hidden: usize,
  out_dim: usize,
  activation: ActivationType,
  out_activation: ActivationType
) -> Vec<Network> {
  let mut networks = Vec::new();

  (0..n).for_each(|i| {
      networks.push(Network::new(format!("net_{}", i) , n_inp, n_hidden, out_dim, activation, out_activation));
  });

  networks
}

#[derive(Clone, Debug)]
pub struct Snake {
  pub head: Point,
  pub body: LinkedList<Point>,
  pub dir: Point,
    pub brain: Network,
    pub score: f64,
    pub speed: f64,
    pub min_dist: f64,
    pub game_over: bool
}

pub struct Enviroment {
  pub agents: Vec<Snake>,
  pub squares: usize,
}

impl Enviroment {
  pub fn new(squares: usize, n: usize) -> Enviroment {
    let init_generatin = create_generation(n, 4, 16, 4, ActivationType::ReLU, ActivationType::Softmax);
    let agents = init_generatin.to_owned().into_iter().map(|net| {
      Snake {
        speed: 0.3,
        head: (0, 0),
        body: LinkedList::new(),
        dir: (1, 0),
        min_dist: 10000.0,
        brain: net,
        score: 0.0,
        game_over: false
      }
    }).collect();
    Enviroment { agents, squares }
  }

  pub async fn run(&mut self) {
    let up = (0, -1);
    let down = (0, 1);
    let right = (1, 0);
    let left = (-1, 0);
    let mut fruit: Point = (rand::gen_range(0, self.squares as i16), rand::gen_range(0, self.squares as i16));
    let mut new_agents: Vec<Snake> = Vec::new();

    for ep in 0..100 {
      for snake in self.agents.iter() {
        let mut new_snake = snake.to_owned();
        let mut j = 0;
        loop {
          let inps = vec![
            (snake.head.0 as f64) / self.squares as f64 * 0.1,
            (snake.head.1 as f64) / self.squares as f64 * 0.1,
            (fruit.0 as f64) / self.squares as f64 * 0.1,
            (fruit.1 as f64) / self.squares as f64 * 0.1
          ];

          let out = argmax(&snake.brain.layer_output(inps.to_owned()).to_vec());

          let curr_dir = match out {
            2 => right,
            3 => left,
            0 => up,
            1 => down,
            _ => (0, 0)
          };
          new_snake.dir = curr_dir;
    
          new_snake.body.push_front(new_snake.head);
          new_snake.head = (new_snake.head.0 + new_snake.dir.0, new_snake.head.1 + new_snake.dir.1);
          let curr_dist: f64 = (((new_snake.head.0 - fruit.0).abs().pow(2) + (new_snake.head.1 - fruit.1).abs().pow(2)) as f64).sqrt();
          
          if curr_dist < new_snake.min_dist {
            new_snake.min_dist = curr_dist;
          }
          if new_snake.head == fruit {
              new_snake.score += 0.1;
              new_snake.speed *= 1.0;
              break;
          } else {
            new_snake.body.pop_back();
          }
          if new_snake.head.0 < 0
              || new_snake.head.1 < 0
              || new_snake.head.0 >= self.squares as i16
              || new_snake.head.1 >= self.squares as i16
          {
              new_snake.game_over = true;
              new_snake.score -= 1.0;
          }
          for (x, y) in &new_snake.body {
              if *x == new_snake.head.0 && *y == new_snake.head.1 {
                new_snake.game_over = true;
              }
          }
          if new_snake.game_over {
            break;
          }
    
          clear_background(LIGHTGRAY);
    
          let game_size = screen_width().min(screen_height());
          let offset_x = (screen_width() - game_size) / 2. + 10.;
          let offset_y = (screen_height() - game_size) / 2. + 10.;
          let sq_size = (screen_height() - offset_y * 2.) / self.squares as f32;
      
          draw_rectangle(offset_x, offset_y, game_size - 20., game_size - 20., WHITE);
      
          for i in 1..self.squares {
              draw_line(
                  offset_x,
                  offset_y + sq_size * i as f32,
                  screen_width() - offset_x,
                  offset_y + sq_size * i as f32,
                  2.,
                  LIGHTGRAY,
              );
          }
      
          for i in 1..self.squares {
              draw_line(
                  offset_x + sq_size * i as f32,
                  offset_y,
                  offset_x + sq_size * i as f32,
                  screen_height() - offset_y,
                  2.,
                  LIGHTGRAY,
              );
          }
    
          draw_rectangle(
              offset_x + new_snake.head.0 as f32 * sq_size,
              offset_y + new_snake.head.1 as f32 * sq_size,
              sq_size,
              sq_size,
              DARKGREEN,
          );
    
          for (x, y) in &new_snake.body {
              draw_rectangle(
                  offset_x + *x as f32 * sq_size,
                  offset_y + *y as f32 * sq_size,
                  sq_size,
                  sq_size,
                  LIME,
              );
          }
    
          draw_rectangle(
              offset_x + fruit.0 as f32 * sq_size,
              offset_y + fruit.1 as f32 * sq_size,
              sq_size,
              sq_size,
              GOLD,
          );
          next_frame().await;
          j += 1;
          if j == 128 {
            break;
          }
        }
        fruit = (rand::gen_range(0, self.squares as i16), rand::gen_range(0, self.squares as i16));
        new_agents.push(new_snake);
      }

      let selected_agents: Vec<Snake> = new_agents.to_owned().into_iter().filter(|ag| ag.score > 0.0).collect();
      let cross_overed_agents = crossing_over(selected_agents.to_owned());
      let mutated = mutation(cross_overed_agents.to_owned(), 0.4);
    }
  }
}
