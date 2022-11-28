pub mod entity;

use glutin_window::GlutinWindow;
use graphics::color::{GREEN, YELLOW};
use opengl_graphics::GlGraphics;
use piston::{WindowSettings, EventSettings, Events, RenderEvent, EventLoop, Event};
use rand::Rng;

use self::entity::{EntityType, Snake, Entity, Direction, FieldInfo};

pub struct Env {
    pub window: GlutinWindow,
    pub events: Events,
    pub entity_type: EntityType,
    pub obj: Snake,
    pub food: [i32; 2],
    step_ratio: f64,
    size: [u32; 2],
    gl: GlGraphics,
}

impl Env {
  pub fn new(size: [u32; 2], step_ratio: f64, entity_type: EntityType) -> Env {

    let window: GlutinWindow = WindowSettings::new("snake env", size)
      .exit_on_esc(true)
      .build()
      .unwrap();

    let events = Events::new(EventSettings::new()).ups(8);

    let gl = GlGraphics::new(opengl_graphics::OpenGL::V3_2);

    let food_x = rand::thread_rng().gen_range(0..size[0]) as i32;
    let food_y = rand::thread_rng().gen_range(0..size[1]) as i32;

    Env { window, events, entity_type, size, step_ratio, obj: Snake::new(step_ratio), gl, food: [food_x, food_y] }
  }

  pub fn render(&mut self, e: Event) {
    if let Some(r) = e.render_args() {
      self.gl.draw(r.viewport(), |_c, gl| {
        graphics::clear(GREEN, gl);

        let square = graphics::rectangle::square(self.food[0] as f64, self.food[1] as f64, 20_f64);

        gl.draw(r.viewport(), |c, gl| {
          let transform = c.transform;

          graphics::rectangle(YELLOW, square, transform, gl);
        });

        self.obj.render(gl, &r);
        self.obj.update();
      });
    }
  }

  pub fn make_action(&mut self, action: Direction) -> FieldInfo {
    let mut reward = 0.0;
    self.obj.update();

    self.obj.step(action.clone());

    if self.obj.pos_x == self.food[0] && self.obj.pos_y == self.food[1] {
      reward += 5.0;
      self.food[0] = rand::thread_rng().gen_range(0..self.size[0]) as i32;
      self.food[1] = rand::thread_rng().gen_range(0..self.size[1]) as i32;
    }

    let curr_dir = match action.clone() {
        Direction::Right => 0,
        Direction::Left => 1,
        Direction::Bottom => 2,
        Direction::Up => 3,
    };

    let food_distance = (((self.food[0] - self.obj.pos_x).pow(2) + (self.food[1] - self.obj.pos_y).pow(2)) as f64).sqrt();
    reward -= food_distance * 10e-3;

    FieldInfo {
      curr_dir,
      obj_pos: [
        self.obj.pos_x,
        self.obj.pos_y
      ],
      food_pos: [self.food[0], self.food[1]],
      food_distance,
      reward
    }
  }
}