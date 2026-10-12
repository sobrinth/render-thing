mod geometry;
mod level;
mod physics;
mod player;

use engine::input::{self as einput, ElementState};
use engine::{CameraView, Engine, MaterialConstants, MaterialPass};
use eros::Context;
use level::{Level, OrbitParams, build_box, build_hex_platform};
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

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const LOOK_SENSITIVITY: f32 = 0.003;
const PHYSICS_DT: f32 = 1.0 / 60.0;
const TARGET_FRAME_TIME: std::time::Duration = std::time::Duration::from_micros(16_000); // ~60 fps

fn main() -> eros::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = GameApp::Starting;
    event_loop.run_app(&mut app)?;
    match app {
        GameApp::Failed(err) => Err(err),
        _ => Ok(()),
    }
}

enum GameApp {
    Starting,
    Running(Game),
    Stopped,
    Failed(eros::ErrorUnion),
}

struct Game {
    engine: Engine,
    ui: egui_winit::State,
    player: Player,
    level: Level,
    held_keys: HashSet<einput::Key>,
    last_frame: Instant,
    accumulator: f32,
    cursor_captured: bool,
    scene_panel: scene::ScenePanel,
    /// Reused across frames to avoid re-allocating the draw list.
    draw_buffer: Vec<engine::DrawCall>,
    window: Window,
}

impl Game {
    fn start(event_loop: &ActiveEventLoop) -> eros::Result<Self> {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Game")
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .context("Creating window")?;

        let window_size = (window.inner_size().width, window.inner_size().height);
        let mut engine = Engine::initialize(
            &window,
            window_size,
            engine::EngineConfig {
                app_name: "Game".to_string(),
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

        let level = match build_level(&mut engine) {
            Ok(level) => level,
            Err(err) => {
                engine.stop();
                return Err(err);
            }
        };

        Ok(Game {
            engine,
            ui,
            player: Player::new(),
            level,
            held_keys: HashSet::new(),
            last_frame: Instant::now(),
            accumulator: 0.0,
            cursor_captured: false,
            scene_panel: scene::ScenePanel::new(),
            draw_buffer: Vec::new(),
            window,
        })
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let _ = self.ui.on_window_event(&self.window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => self.engine.resize(size.into()),

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
                    let _ = self.window.set_cursor_grab(CursorGrabMode::None);
                    self.window.set_cursor_visible(true);
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
                self.engine.on_key_press((state, key));
            }

            WindowEvent::MouseInput {
                button: winit::event::MouseButton::Left,
                state: winit::event::ElementState::Pressed,
                ..
            } if !self.cursor_captured && !self.engine.wants_pointer_input() => {
                let _ = self
                    .window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
                self.window.set_cursor_visible(false);
                self.cursor_captured = true;
            }

            WindowEvent::RedrawRequested => self.render(),

            _ => {}
        }
    }

    fn device_event(&mut self, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.cursor_captured
        {
            self.player.yaw += delta.0 as f32 * LOOK_SENSITIVITY;
            self.player.pitch -= delta.1 as f32 * LOOK_SENSITIVITY;
            self.player.pitch = self
                .player
                .pitch
                .clamp(-89f32.to_radians(), 89f32.to_radians());
        }
    }

    fn render(&mut self) {
        let now = Instant::now();
        let frame_dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // Fixed-timestep physics (60 TPS)
        self.accumulator += frame_dt;
        while self.accumulator >= PHYSICS_DT {
            self.player.update(PHYSICS_DT, &self.held_keys);
            self.level.tick(PHYSICS_DT);
            self.player.resolve_collision(&self.level.collision_boxes);
            self.accumulator -= PHYSICS_DT;
        }

        // Build camera from current player state
        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height as f32;
        let mut proj = glm::perspective_rh_zo(aspect, 70f32.to_radians(), 10000f32, 0.1f32);
        proj[(1, 1)] *= -1.0;
        let camera = CameraView {
            view_matrix: self.player.view_matrix(),
            proj_matrix: proj,
            position: self.player.eye_position(),
        };

        // Render
        let raw_input = self.ui.take_egui_input(&self.window);
        self.window.pre_present_notify();
        self.level.all_draws_into(&mut self.draw_buffer);
        let scene_panel = &mut self.scene_panel;
        let level = &mut self.level;
        let platform_output = self
            .engine
            .draw(camera, &self.draw_buffer, raw_input, |ctx| {
                scene_panel.show(ctx, &mut level.scene);
            });
        self.ui
            .handle_platform_output(&self.window, platform_output);

        let elapsed = now.elapsed();
        if elapsed < TARGET_FRAME_TIME {
            std::thread::sleep(TARGET_FRAME_TIME - elapsed);
        }

        self.window.request_redraw();
    }
}

impl ApplicationHandler for GameApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let GameApp::Starting = self {
            *self = match Game::start(event_loop) {
                Ok(game) => GameApp::Running(game),
                Err(err) => {
                    event_loop.exit();
                    GameApp::Failed(err)
                }
            };
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let GameApp::Running(game) = self {
            game.window_event(event_loop, event);
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let GameApp::Running(game) = self {
            game.device_event(event);
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        if let GameApp::Running(game) = self {
            game.engine.stop();
            *self = GameApp::Stopped;
        }
    }
}

fn build_level(engine: &mut Engine) -> eros::Result<Level> {
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
    let box_mesh = engine.upload_mesh(&cube_indices, &cube_vertices)?;

    let yellow_material = engine.create_material_colored(
        MaterialConstants {
            color_factors: [1.0, 0.6, 0.0, 1.0],
            metal_rough_factors: [0.3, 1.0, 0.0, 0.0],
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

    let floor_root = scene.add_root("floor", glm::Mat4::identity());

    let (_, floor_aabb) = build_box(
        &mut scene,
        floor_root,
        "floor-mesh",
        box_mesh,
        yellow_material,
        glm::vec3(-50.0, -0.5, -50.0),
        glm::vec3(50.0, 0.0, 50.0),
    );
    collision_boxes.push(floor_aabb);

    let proc_root = scene.add_child(floor_root, "procedural", glm::Mat4::identity());

    // --- Hex platform chain ---
    // Each mesh is generated at its final size to avoid non-uniform scaling
    // (which would corrupt normals without inverse-transpose correction).
    let (verts, idx) = geometry::hex_prism_mesh(4.0, 0.5);
    let hex_mesh_1 = engine.upload_mesh(&idx, &verts)?;

    let (verts, idx) = geometry::hex_prism_mesh(3.0, 0.5);
    let hex_mesh_2 = engine.upload_mesh(&idx, &verts)?;

    let (verts, idx) = geometry::hex_prism_mesh(2.0, 0.5);
    let hex_mesh_3 = engine.upload_mesh(&idx, &verts)?;

    let p1 = build_hex_platform(
        &mut scene,
        proc_root,
        "platform-1",
        hex_mesh_1,
        checkerboard_material,
        glm::vec3(0.0, 1.25, -12.5),
    );
    let t = scene.global_transform(p1);
    collision_boxes.push(geometry::hex_aabb(
        glm::vec3(t[(0, 3)], t[(1, 3)], t[(2, 3)]),
        4.0,
        0.5,
    ));

    let p2 = build_hex_platform(
        &mut scene,
        p1,
        "platform-2",
        hex_mesh_2,
        yellow_material,
        glm::vec3(0.0, 1.0, -6.0),
    );
    let t = scene.global_transform(p2);
    let p2_aabb_idx = collision_boxes.len();
    collision_boxes.push(geometry::hex_aabb(
        glm::vec3(t[(0, 3)], t[(1, 3)], t[(2, 3)]),
        3.0,
        0.5,
    ));

    let p3 = build_hex_platform(
        &mut scene,
        p2,
        "platform-3",
        hex_mesh_3,
        checkerboard_material,
        glm::vec3(0.0, 1.5, -6.0),
    );
    let t = scene.global_transform(p3);
    let p3_aabb_idx = collision_boxes.len();
    collision_boxes.push(geometry::hex_aabb(
        glm::vec3(t[(0, 3)], t[(1, 3)], t[(2, 3)]),
        2.0,
        0.5,
    ));

    let world = glm::rotate(
        &glm::translate(&glm::Mat4::identity(), &glm::vec3(-25.0, 0.5, 0.0)),
        90.0f32.to_radians(),
        &glm::vec3(0.0, 1.0, 0.0),
    );
    match scene::load_gltf(engine, "assets/models/downloaded/sponza/Sponza.gltf") {
        Ok(gltf) => {
            let sponza_root = scene.add_child(floor_root, "sponza", world);
            scene.adopt(sponza_root, gltf);
        }
        Err(err) => log::error!("Continuing without the Sponza scene: {err:?}"),
    }

    let mut level = Level::new(scene, collision_boxes);
    level.add_orbit(
        p2,
        p2_aabb_idx,
        OrbitParams {
            orbit_radius: 6.0,
            y_offset: 1.0,
            period_secs: 10.0,
            mesh_radius: 3.0,
            mesh_height: 0.5,
        },
    );
    level.add_orbit(
        p3,
        p3_aabb_idx,
        OrbitParams {
            orbit_radius: 6.0,
            y_offset: 1.5,
            period_secs: 5.0,
            mesh_radius: 2.0,
            mesh_height: 0.5,
        },
    );
    Ok(level)
}
