use engine::Engine;
use engine::input::{self as einput};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{MouseScrollDelta, StartCause, WindowEvent};
use winit::event_loop::ControlFlow::Poll;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
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
    ui: Option<egui_winit::State>,
}

impl ApplicationHandler for Application {
    fn new_events(&mut self, _: &ActiveEventLoop, _: StartCause) {}

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Render Thing")
                    .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        let window_size = (window.inner_size().width, window.inner_size().height);
        let engine = Engine::initialize(&window, window_size);

        let ui = egui_winit::State::new(
            engine.egui_context(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            Some(winit::window::Theme::Dark),
            None,
        );

        self.engine = Some(engine);
        self.ui = Some(ui);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = self.window.as_ref().unwrap();
        let ui = self.ui.as_mut().unwrap();
        let engine = self.engine.as_mut().unwrap();

        let _ = ui.on_window_event(window, &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                engine.resize(size.into());
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
                engine.on_mouse_button_event(button, state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                engine.on_mouse_event(position.into());
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
                        winit::keyboard::NamedKey::F3 => einput::NamedKey::F3,
                        _ => einput::NamedKey::Other,
                    }),
                    _ => einput::Key::Other,
                };
                engine.on_key_press((state, key));
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, _v_lines),
                ..
            } => {}
            WindowEvent::RedrawRequested => {
                let raw_input = ui.take_egui_input(window);
                window.pre_present_notify();
                let platform_output = engine.draw(raw_input);
                ui.handle_platform_output(window, platform_output);
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
