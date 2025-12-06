use engine::Engine;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{MouseScrollDelta, StartCause, WindowEvent};
use winit::event_loop::ControlFlow::Poll;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(Poll);

    let mut app = Application::default();
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
struct Application {
    window: Option<Window>,
    engine: Option<Engine>,
}

impl ApplicationHandler for Application {
    fn new_events(&mut self, _: &ActiveEventLoop, _: StartCause) {
        if let Some(_app) = self.engine.as_mut() {
            // app.wheel_delta = None;
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Render Thing")
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        self.engine = Some(Engine::initialize(&window));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let engine = self.engine.as_mut().unwrap();

        engine.on_window_event(self.window.as_ref().unwrap(), &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                let app = self.engine.as_mut().unwrap();
                app.resize(size.into());
            }
            WindowEvent::MouseInput { .. } => {
                // self.vulkan.as_mut().unwrap().is_left_clicked =
                //     state == ElementState::Pressed && button == MouseButton::Left;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let _app = self.engine.as_mut().unwrap();

                let _position: (i32, i32) = position.into();
                // app.cursor_delta = Some([
                //     app.cursor_position[0] - position.0,
                //     app.cursor_position[1] - position.1,
                // ]);
                // app.cursor_position = [position.0, position.1];
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, _v_lines),
                ..
            } => {
                // self.vulkan.as_mut().unwrap().wheel_delta = Some(v_lines);
            }
            WindowEvent::RedrawRequested => {
                let app = self.engine.as_mut().unwrap();
                let window = self.window.as_ref().unwrap();

                window.pre_present_notify();

                app.draw(window);

                std::thread::sleep(std::time::Duration::from_millis(1000 / 60)); // not really 60 fps
                window.request_redraw();
            }
            _ => (),
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        // This is guaranteed to be the last code run before the event-loop will exit, so the engine
        // is dropped here because it needs to be gone before the event-loop exits.
        let mut engine = self.engine.take().unwrap();
        engine.stop();
        // This drop is not needed but makes the intention here explicit
        drop(engine)
    }
}
