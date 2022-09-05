use rand::Rng;
use sdl2::Sdl;
use sdl2::pixels::{Color, PixelFormatEnum};
use sdl2::rect::{Point, Rect};
use sdl2::render::WindowCanvas;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use std::time::Duration;

use crate::network::Network;

pub struct Agent {
  pub brain: Network,
  pub color: Color,
  pub position: Point,
  pub rect: Rect,
}

pub struct Enviroment {
  pub width: usize,
  pub bg_color: Color,
  pub height: usize,
  pub canvas: WindowCanvas,
  pub context: Sdl,
  pub agents: Vec<Agent>
}

impl Enviroment {
  pub fn new(width: usize, height: usize, bg_color: Color, generation: Vec<Network>, agent_width: usize, agent_height: usize) -> Enviroment {
    let context = sdl2::init().unwrap();
    let video_subsystem = context.video().unwrap();

    let window = video_subsystem.window("enviroment", width as u32, height as u32)
        .position_centered()
        .build()
        .expect("could not initialize video subsystem");

    let canvas = window.into_canvas().build()
        .expect("could not make a canvas");

    let agents = generation.into_iter().map(|net| {
      let pos_x = rand::thread_rng().gen_range(-20..20) as i32;
      let pos_y = rand::thread_rng().gen_range(-20..20) as i32;
      let rect = Rect::new(pos_x, pos_y, agent_width as u32, agent_height as u32);
  
      return Agent {
        brain: net,
        rect: rect,
        position: Point::new(
          pos_x, 
          pos_y,
        ),
        color: Color::RGB(
          rand::thread_rng().gen_range(0..=255), 
          rand::thread_rng().gen_range(0..=255),
          rand::thread_rng().gen_range(0..=255))
        }
    }).collect();

    Enviroment {
      width,
      agents,
      bg_color,
      context,
      height,
      canvas
    }
  }

  fn render(&mut self) {
    self.canvas.set_draw_color(self.bg_color);
    self.canvas.clear();

    let texture_creator = self.canvas.texture_creator();

    self.agents.iter().for_each(|agent| {
      let screen_position = agent.position + Point::new(self.width as i32 / 2, self.height as i32 / 2);
      let screen_rect = Rect::from_center(screen_position, agent.rect.width(), agent.rect.height());

      let texture = texture_creator.create_texture(
        PixelFormatEnum::ABGR1555,
        sdl2::render::TextureAccess::Target,
        agent.rect.width(),
        agent.rect.height()
      ).unwrap();

      self.canvas.copy(&texture, agent.rect, screen_rect).expect("error to copy");
    });

    self.canvas.present();
  }

  pub fn run(&mut self) {
    let mut event_pump = self.context.event_pump().unwrap();
    let mut i = 0;
  
    'running: loop {
        // Handle events
        for event in event_pump.poll_iter() {
          match event {
            Event::Quit {..} |
            Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                break 'running;
            },
            Event::KeyDown { keycode: Some(Keycode::Left), .. } => {
              self.agents[0].position = self.agents[0].position.offset(-2, 0);
            },
            Event::KeyDown { keycode: Some(Keycode::Right), .. } => {
              self.agents[0].position = self.agents[0].position.offset(2, 0);
            },
            Event::KeyDown { keycode: Some(Keycode::Up), .. } => {
              self.agents[0].position = self.agents[0].position.offset(0, -2);
            },
            Event::KeyDown { keycode: Some(Keycode::Down), .. } => {
              self.agents[0].position = self.agents[0].position.offset(0, 2);
            },
            _ => {}
          }
          println!("pos1: {:?}", self.agents[0].position);
        }

        // Update
        i = (i + 1) % 255;

        // Render
        self.render();

        // Time management!
        std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
  }
  // pub fn render(&mut self) {
  //   self.canvas.set_draw_color(self.color);
  //   self.canvas.clear();
  //   self.canvas.present();
  // }
}



