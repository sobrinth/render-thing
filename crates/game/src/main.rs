mod level;
mod physics;
mod player;

use engine::input::{self as einput, ElementState};
use engine::{CameraView, Engine, MaterialConstants, MaterialPass};
use level::{Level, build_box};
use nalgebra_glm as glm;
use player::Player;
use scene::SceneGraph;
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
const TARGET_FRAME_TIME: std::time::Duration = std::time::Duration::from_micros(16_000); // ~60 fps

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = GameApp::default();
    event_loop.run_app(&mut app).unwrap();
}

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
    scene_panel: scene::ScenePanel,
}

impl Default for GameApp {
    fn default() -> Self {
        Self {
            window: None,
            engine: None,
            ui: None,
            player: None,
            level: None,
            held_keys: HashSet::new(),
            last_frame: None,
            accumulator: 0.0,
            cursor_captured: false,
            scene_panel: scene::ScenePanel::new(),
        }
    }
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

        let window_size = (window.inner_size().width, window.inner_size().height);
        let mut engine = Engine::initialize(
            &window,
            window_size,
            engine::EngineConfig {
                app_name: "Game".to_string(),
                ..Default::default()
            },
        );

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
        self.cursor_captured = false;
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.cursor_captured
            && let Some(player) = &mut self.player
        {
            player.yaw += delta.0 as f32 * LOOK_SENSITIVITY;
            player.pitch -= delta.1 as f32 * LOOK_SENSITIVITY;
            player.pitch = player.pitch.clamp(-89f32.to_radians(), 89f32.to_radians());
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
                        winit::keyboard::NamedKey::F4 => einput::NamedKey::F4,
                        _ => einput::NamedKey::Other,
                    }),
                    _ => einput::Key::Other,
                };
                if state == ElementState::Pressed {
                    self.held_keys.insert(key.clone());
                } else {
                    self.held_keys.remove(&key);
                }
                engine.on_key_press((state, key));
            }

            WindowEvent::MouseInput {
                button: winit::event::MouseButton::Left,
                state: winit::event::ElementState::Pressed,
                ..
            } if !self.cursor_captured && !engine.egui_context().wants_pointer_input() => {
                let _ = window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                window.set_cursor_visible(false);
                self.cursor_captured = true;
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
            if let Some(level) = &self.level {
                player.resolve_collision(&level.collision_boxes);
            }
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
        window.pre_present_notify();
        let draws = self
            .level
            .as_ref()
            .map(|l| l.all_draws())
            .unwrap_or_default();
        let scene_panel = &mut self.scene_panel;
        let level = self.level.as_mut();
        let platform_output = engine.draw(camera, &draws, raw_input, |ctx| {
            if let Some(l) = level {
                scene_panel.show(ctx, &mut l.scene);
            }
        });
        ui.handle_platform_output(window, platform_output);

        let elapsed = now.elapsed();
        if elapsed < TARGET_FRAME_TIME {
            std::thread::sleep(TARGET_FRAME_TIME - elapsed);
        }

        window.request_redraw();
    }
}

fn build_level(engine: &mut Engine) -> Level {
    let cube_vertices = vec![
        // -Z face (normal [0, 0, -1])
        engine::Vertex {
            position: [-0.5, -0.5, -0.5],
            normal: [0.0, 0.0, -1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [0.5, -0.5, -0.5],
            normal: [0.0, 0.0, -1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [0.5, 0.5, -0.5],
            normal: [0.0, 0.0, -1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 0.0,
        },
        engine::Vertex {
            position: [-0.5, 0.5, -0.5],
            normal: [0.0, 0.0, -1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        },
        // +Z face (normal [0, 0, 1])
        engine::Vertex {
            position: [0.5, -0.5, 0.5],
            normal: [0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [-0.5, -0.5, 0.5],
            normal: [0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [-0.5, 0.5, 0.5],
            normal: [0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 0.0,
        },
        engine::Vertex {
            position: [0.5, 0.5, 0.5],
            normal: [0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        },
        // -X face (normal [-1, 0, 0])
        engine::Vertex {
            position: [-0.5, -0.5, 0.5],
            normal: [-1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [-0.5, -0.5, -0.5],
            normal: [-1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [-0.5, 0.5, -0.5],
            normal: [-1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 0.0,
        },
        engine::Vertex {
            position: [-0.5, 0.5, 0.5],
            normal: [-1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        },
        // +X face (normal [1, 0, 0])
        engine::Vertex {
            position: [0.5, -0.5, -0.5],
            normal: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [0.5, -0.5, 0.5],
            normal: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [0.5, 0.5, 0.5],
            normal: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 0.0,
        },
        engine::Vertex {
            position: [0.5, 0.5, -0.5],
            normal: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        },
        // -Y face (normal [0, -1, 0])
        engine::Vertex {
            position: [-0.5, -0.5, 0.5],
            normal: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [0.5, -0.5, 0.5],
            normal: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [0.5, -0.5, -0.5],
            normal: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 0.0,
        },
        engine::Vertex {
            position: [-0.5, -0.5, -0.5],
            normal: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        },
        // +Y face (normal [0, 1, 0])
        engine::Vertex {
            position: [-0.5, 0.5, -0.5],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [0.5, 0.5, -0.5],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 1.0,
        },
        engine::Vertex {
            position: [0.5, 0.5, 0.5],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 1.0,
            uv_y: 0.0,
        },
        engine::Vertex {
            position: [-0.5, 0.5, 0.5],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        },
    ];
    let cube_indices: Vec<u32> = vec![
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17,
        18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    ];
    let box_mesh = engine.upload_mesh(&cube_indices, &cube_vertices);

    let yellow_material = engine.create_material_colored(
        MaterialConstants {
            color_factors: [1.0, 0.6, 0.0, 1.0],
            metal_rough_factors: [0.3, 1.0, 0.0, 0.0],
            ..Default::default()
        },
        MaterialPass::MainColor,
    );
    let checkerboard_material = engine.create_material(
        engine.checkerboard_texture(),
        engine.metal_rough_texture(),
        MaterialConstants::default(),
        MaterialPass::MainColor,
    );

    let mut scene = SceneGraph::new();
    let mut collision_boxes = Vec::new();

    let proc_root = scene.add_root("procedural", glm::Mat4::identity());

    {
        let mut add_box = |name: &str, min: glm::Vec3, max: glm::Vec3, material: engine::MaterialHandle| {
            let (_, aabb) = build_box(&mut scene, proc_root, name, box_mesh, material, min, max);
            collision_boxes.push(aabb);
        };
        add_box(
            "floor",
            glm::vec3(-50.0, -0.5, -50.0),
            glm::vec3(50.0, 0.0, 50.0),
            yellow_material,
        );
        add_box(
            "platform-1",
            glm::vec3(-4.0, 1.0, -15.0),
            glm::vec3(4.0, 1.5, -10.0),
            checkerboard_material,
        );
        add_box(
            "platform-2",
            glm::vec3(-3.0, 2.0, -21.0),
            glm::vec3(3.0, 2.5, -16.0),
            yellow_material,
        );
        add_box(
            "platform-3",
            glm::vec3(-2.0, 3.5, -27.0),
            glm::vec3(2.0, 4.0, -22.0),
            checkerboard_material,
        );
    }

    let world = glm::rotate(
        &glm::translate(&glm::Mat4::identity(), &glm::vec3(-25.0, 0.5, 0.0)),
        90.0f32.to_radians(),
        &glm::vec3(0.0, 1.0, 0.0),
    );
    if let Some(gltf) = scene::load_gltf(engine, "assets/models/downloaded/sponza/Sponza.gltf") {
        let sponza_root = scene.add_root("sponza", world);
        scene.adopt(sponza_root, gltf);
    }

    Level::new(scene, collision_boxes)
}
