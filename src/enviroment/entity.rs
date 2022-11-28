use graphics::color::RED;
use opengl_graphics::GlGraphics;
use piston::RenderArgs;

#[derive(Clone, Debug)]
pub enum Direction {
  Right,
  Left,
  Bottom, 
  Up
}

#[derive(Debug)]
pub struct FieldInfo {
  pub curr_dir: u8,
  pub obj_pos: [i32; 2],
  pub food_pos: [i32; 2],
  pub food_distance: f64,
  pub reward: f64,
}

pub enum EntityType {
  Snake,
}

pub trait Entity {
    fn new(step_ratio: f64) -> Self;
    fn render(&self, gl: &mut GlGraphics, args: &RenderArgs);
    fn update(&mut self);
    fn step(&mut self, direction: Direction);
}

#[derive(Clone)]
pub struct Snake {
    pub pos_x: i32,
    pub pos_y: i32,
    pub direction: Direction,
    step_ratio: f64
}

impl Entity for Snake {
    fn new(step_ratio: f64) -> Self {
      Snake { pos_x: 30, pos_y: 30, direction: Direction::Right, step_ratio }
    }

    fn render(&self, gl: &mut GlGraphics, args: &RenderArgs) {
        let square = graphics::rectangle::square(self.pos_x as f64, self.pos_y as f64, 20_f64);

        gl.draw(args.viewport(), |c, gl| {
          let transform = c.transform;

          graphics::rectangle(RED, square, transform, gl);
        })
    }

    fn update(&mut self) {
        match self.direction {
            Direction::Right => self.pos_x = (self.pos_x as f64 + self.step_ratio) as i32,
            Direction::Left => self.pos_x = (self.pos_x as f64 - self.step_ratio) as i32,
            Direction::Bottom => self.pos_y = (self.pos_y as f64 - self.step_ratio) as i32,
            Direction::Up => self.pos_y = (self.pos_y as f64 + self.step_ratio) as i32,
        }
    }

    fn step(&mut self, direction: Direction) {
      match direction {
        Direction::Right => self.pos_x = (self.pos_x as f64 + self.step_ratio) as i32,
        Direction::Left => self.pos_x = (self.pos_x as f64 - self.step_ratio) as i32,
        Direction::Bottom => self.pos_y = (self.pos_y as f64 - self.step_ratio) as i32,
        Direction::Up => self.pos_y = (self.pos_y as f64 + self.step_ratio) as i32,
      }
  }
}