use engine::input::{self as einput};
use engine::{CameraView, Engine};
use eros::Context;
use nalgebra_glm as glm;
use winit::application::ApplicationHandler;

mod camera;
use camera::Camera;
use winit::dpi::PhysicalSize;
use winit::event::{MouseScrollDelta, WindowEvent};
use winit::event_loop::ControlFlow::Poll;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::window::{Window, WindowId};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn main() -> eros::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(Poll);

    let mut app = Application::Starting;
    event_loop.run_app(&mut app)?;
    match app {
        Application::Failed(err) => Err(err),
        _ => Ok(()),
    }
}

enum Application {
    Starting,
    Running(Viewer),
    Stopped,
    Failed(eros::ErrorUnion),
}

struct Viewer {
    engine: Engine,
    ui: egui_winit::State,
    camera: Camera,
    mouse_pos: (i32, i32),
    scene: Option<scene::SceneGraph>,
    scene_panel: scene::ScenePanel,
    /// Reused across frames to avoid re-allocating the draw list.
    draw_buffer: Vec<engine::DrawCall>,
    window: Window,
}

impl Viewer {
    fn start(event_loop: &ActiveEventLoop) -> eros::Result<Self> {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Render Thing")
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .context("Creating window")?;

        let window_size = (window.inner_size().width, window.inner_size().height);
        let mut engine = Engine::initialize(
            &window,
            window_size,
            engine::EngineConfig {
                app_name: "Render Thing".to_string(),
                ..Default::default()
            },
        )?;

        let ui = egui_winit::State::new(
            engine.egui_context(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            Some(winit::window::Theme::Dark),
            None,
        );

        let scene = match scene::load_gltf(&mut engine, "assets/models/downloaded/abbey.glb") {
            Ok(scene) => Some(scene),
            Err(err) => {
                log::error!("Starting without a scene: {err:?}");
                None
            }
        };

        Ok(Viewer {
            engine,
            ui,
            camera: Camera::new(),
            mouse_pos: (0, 0),
            scene,
            scene_panel: scene::ScenePanel::new(),
            draw_buffer: Vec::new(),
            window,
        })
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let _ = self.ui.on_window_event(&self.window, &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.engine.resize(size.into());
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let button = match button {
                    winit::event::MouseButton::Left => einput::MouseButton::Left,
                    winit::event::MouseButton::Right => einput::MouseButton::Right,
                    winit::event::MouseButton::Middle => einput::MouseButton::Middle,
                    _ => einput::MouseButton::Other,
                };
                let state = match state {
                    winit::event::ElementState::Pressed => einput::ElementState::Pressed,
                    winit::event::ElementState::Released => einput::ElementState::Released,
                };
                if !self.engine.wants_pointer_input() {
                    self.camera.handle_mouse_button_event(button, state);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos: (i32, i32) = position.into();
                if !self.engine.wants_pointer_input() {
                    self.camera.handle_mouse_event(self.mouse_pos, new_pos);
                }
                self.mouse_pos = new_pos;
            }
            WindowEvent::KeyboardInput { event, .. } if !event.repeat => {
                let state = match event.state {
                    winit::event::ElementState::Pressed => einput::ElementState::Pressed,
                    winit::event::ElementState::Released => einput::ElementState::Released,
                };
                let key = match event.key_without_modifiers() {
                    winit::keyboard::Key::Character(c) => einput::Key::Character(c.to_string()),
                    winit::keyboard::Key::Named(n) => einput::Key::Named(match n {
                        winit::keyboard::NamedKey::Space => einput::NamedKey::Space,
                        winit::keyboard::NamedKey::Shift => einput::NamedKey::Shift,
                        winit::keyboard::NamedKey::F2 => einput::NamedKey::F2,
                        winit::keyboard::NamedKey::F3 => einput::NamedKey::F3,
                        winit::keyboard::NamedKey::F4 => einput::NamedKey::F4,
                        _ => einput::NamedKey::Other,
                    }),
                    _ => einput::Key::Other,
                };
                self.engine.on_key_press((state, key.clone()));
                if !self.engine.egui_context().egui_wants_keyboard_input() {
                    self.camera.handle_key_event((state, key));
                }
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, _v_lines),
                ..
            } => {}
            WindowEvent::RedrawRequested => {
                let raw_input = self.ui.take_egui_input(&self.window);
                self.window.pre_present_notify();
                self.camera.update();
                let view = self.camera.get_view_matrix();
                let aspect =
                    self.window.inner_size().width as f32 / self.window.inner_size().height as f32;
                let mut proj = glm::perspective_rh_zo(aspect, 70f32.to_radians(), 10000f32, 0.1f32);
                proj[(1, 1)] *= -1.0;
                let camera_view = CameraView {
                    view_matrix: view,
                    proj_matrix: proj,
                    position: self.camera.position,
                };
                match &self.scene {
                    Some(scene) => scene.flatten_visible_into(&mut self.draw_buffer),
                    None => self.draw_buffer.clear(),
                }
                let scene_panel = &mut self.scene_panel;
                let scene_opt = &mut self.scene;
                let platform_output =
                    self.engine
                        .draw(camera_view, &self.draw_buffer, raw_input, |ctx| {
                            if let Some(scene) = scene_opt {
                                scene_panel.show(ctx, scene);
                            }
                        });
                self.ui
                    .handle_platform_output(&self.window, platform_output);
                self.window.request_redraw();
            }
            _ => (),
        }
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Application::Starting = self {
            *self = match Viewer::start(event_loop) {
                Ok(viewer) => Application::Running(viewer),
                Err(err) => {
                    event_loop.exit();
                    Application::Failed(err)
                }
            };
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Application::Running(viewer) = self {
            viewer.window_event(event_loop, event);
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        // This is guaranteed to be the last code run before the event-loop will exit, so the
        // viewer (and with it the engine) is dropped here, before the event-loop goes away.
        if let Application::Running(viewer) = self {
            viewer.engine.stop();
            *self = Application::Stopped;
        }
    }
}
