mod level;
mod physics;
mod player;

use engine::input::{self as einput, ElementState};
use engine::{CameraView, Engine};
use level::Level;
use nalgebra_glm as glm;
use player::Player;
use std::collections::HashSet;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowId};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const LOOK_SENSITIVITY: f32 = 0.003;
const PHYSICS_DT: f32 = 1.0 / 60.0;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = GameApp::default();
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
struct GameApp {
    window: Option<Window>,
    engine: Option<Engine>,
    ui: Option<egui_winit::State>,
    player: Option<Player>,
    level: Option<Level>,
    held_keys: HashSet<einput::Key>,
    last_frame: Option<Instant>,
    accumulator: f32,
    cursor_captured: bool,
}

impl ApplicationHandler for GameApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Game")
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        // Capture cursor for first-person look
        let _ = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        window.set_cursor_visible(false);

        let window_size = (window.inner_size().width, window.inner_size().height);
        let mut engine = Engine::initialize(&window, window_size);

        let ui = egui_winit::State::new(
            engine.egui_context(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            Some(winit::window::Theme::Dark),
            None,
        );

        let level = build_level(&mut engine);

        self.engine = Some(engine);
        self.ui = Some(ui);
        self.window = Some(window);
        self.player = Some(Player::new());
        self.level = Some(level);
        self.last_frame = Some(Instant::now());
        self.cursor_captured = true;
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.cursor_captured {
                if let Some(player) = &mut self.player {
                    player.yaw += delta.0 as f32 * LOOK_SENSITIVITY;
                    player.pitch -= delta.1 as f32 * LOOK_SENSITIVITY;
                    player.pitch = player.pitch.clamp(-89f32.to_radians(), 89f32.to_radians());
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if self.engine.is_none() {
            return;
        }
        let window = self.window.as_ref().unwrap();
        let ui = self.ui.as_mut().unwrap();
        let engine = self.engine.as_mut().unwrap();

        let _ = ui.on_window_event(window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => engine.resize(size.into()),

            WindowEvent::KeyboardInput {
                event: key_event, ..
            } if !key_event.repeat => {
                use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;

                // Escape releases cursor; don't record it as a held key
                if matches!(
                    key_event.key_without_modifiers(),
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape)
                ) && key_event.state == winit::event::ElementState::Pressed
                {
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                    self.cursor_captured = false;
                    return;
                }

                let state = match key_event.state {
                    winit::event::ElementState::Pressed => ElementState::Pressed,
                    winit::event::ElementState::Released => ElementState::Released,
                };
                let key = match key_event.key_without_modifiers() {
                    winit::keyboard::Key::Character(c) => einput::Key::Character(c.to_string()),
                    winit::keyboard::Key::Named(n) => einput::Key::Named(match n {
                        winit::keyboard::NamedKey::Space => einput::NamedKey::Space,
                        winit::keyboard::NamedKey::Shift => einput::NamedKey::Shift,
                        winit::keyboard::NamedKey::F2 => einput::NamedKey::F2,
                        winit::keyboard::NamedKey::F3 => einput::NamedKey::F3,
                        _ => einput::NamedKey::Other,
                    }),
                    _ => einput::Key::Other,
                };
                engine.on_key_press((state, key.clone()));
                match state {
                    ElementState::Pressed => {
                        self.held_keys.insert(key);
                    }
                    ElementState::Released => {
                        self.held_keys.remove(&key);
                    }
                }
            }

            WindowEvent::MouseInput {
                button: winit::event::MouseButton::Left,
                state: winit::event::ElementState::Pressed,
                ..
            } => {
                // Recapture cursor when clicking the window after Escape
                if !self.cursor_captured {
                    let _ = window
                        .set_cursor_grab(CursorGrabMode::Locked)
                        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                    window.set_cursor_visible(false);
                    self.cursor_captured = true;
                }
            }

            WindowEvent::RedrawRequested => self.render(),

            _ => {}
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        let mut engine = self.engine.take().unwrap();
        engine.stop();
        drop(engine);
    }
}

impl GameApp {
    fn render(&mut self) {
        let window = self.window.as_ref().unwrap();
        let ui = self.ui.as_mut().unwrap();
        let engine = self.engine.as_mut().unwrap();
        let player = self.player.as_mut().unwrap();
        let level = self.level.as_ref().unwrap();

        // Delta time — cap at 0.1 s to avoid spiral of death on long frames
        let now = Instant::now();
        let frame_dt = self
            .last_frame
            .replace(now)
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0)
            .min(0.1);

        // Fixed-timestep physics (60 TPS)
        self.accumulator += frame_dt;
        while self.accumulator >= PHYSICS_DT {
            player.update(PHYSICS_DT, &self.held_keys);
            player.resolve_collision(&level.collision_boxes);
            self.accumulator -= PHYSICS_DT;
        }

        // Build camera from current player state
        let aspect = window.inner_size().width as f32 / window.inner_size().height as f32;
        let mut proj = glm::perspective_rh_zo(aspect, 70f32.to_radians(), 10000f32, 0.1f32);
        proj[(1, 1)] *= -1.0;
        let camera = CameraView {
            view_matrix: player.view_matrix(),
            proj_matrix: proj,
            position: player.eye_position(),
        };

        // Render
        let raw_input = ui.take_egui_input(window);
        let draws = level.all_draws();
        let platform_output = engine.draw(camera, &draws, raw_input);
        window.pre_present_notify();
        ui.handle_platform_output(window, platform_output);
        window.request_redraw();
    }
}

fn build_level(_engine: &mut Engine) -> Level {
    Level::new(None, vec![], vec![], vec![])
}
