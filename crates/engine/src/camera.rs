use glm::Vec3;
use winit::event::{ElementState, MouseButton};
use winit::keyboard::{Key, NamedKey};

pub struct Camera {
    pub velocity: Vec3,
    pub position: Vec3,
    pitch: f32,
    yaw: f32,
    mouse_pressed: bool,
}

impl Camera {
    pub(crate) fn handle_mouse_event(&mut self, old_pos: (i32, i32), new_pos: (i32, i32)) {
        let delta_x = new_pos.0 - old_pos.0;
        let delta_y = new_pos.1 - old_pos.1;

        if self.mouse_pressed {
            self.yaw += delta_x as f32 / 200f32;
            self.pitch -= delta_y as f32 / 200f32;
        }
    }

    pub(crate) fn handle_mouse_button_event(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.mouse_pressed = state == ElementState::Pressed;
        }
    }

    pub(crate) fn handle_key_event(&mut self, key_event: (ElementState, Key)) {
        match key_event.1 {
            Key::Character(c) => match key_event.0 {
                ElementState::Pressed => match c.as_str() {
                    "w" => self.velocity.z = -1f32,
                    "s" => self.velocity.z = 1f32,
                    "a" => self.velocity.x = -1f32,
                    "d" => self.velocity.x = 1f32,
                    _ => {}
                },
                ElementState::Released => match c.as_str() {
                    "w" => self.velocity.z = 0f32,
                    "s" => self.velocity.z = 0f32,
                    "a" => self.velocity.x = 0f32,
                    "d" => self.velocity.x = 0f32,
                    _ => {}
                },
            },
            Key::Named(n) => match key_event.0 {
                ElementState::Pressed => match n {
                    NamedKey::Space => self.velocity.y = 1f32,
                    NamedKey::Shift => self.velocity.y = -1f32,
                    _ => {}
                },
                ElementState::Released => match n {
                    NamedKey::Space => self.velocity.y = 0f32,
                    NamedKey::Shift => self.velocity.y = 0f32,
                    _ => {}
                },
            },
            Key::Unidentified(_) => {}
            Key::Dead(_) => {}
        }
    }

    pub fn new() -> Self {
        Self {
            velocity: glm::vec3(0.0, 0.0, 0.0),
            position: glm::vec3(0.0, 0.0, 5.0),
            pitch: 0.0,
            yaw: 0.0,
            mouse_pressed: false,
        }
    }

    pub fn update(&mut self) {
        let camera_rotation = self.get_rotation_matrix();
        let movement = camera_rotation * glm::vec3_to_vec4(&self.velocity);
        self.position += glm::vec4_to_vec3(&movement);
    }

    pub fn get_view_matrix(&self) -> glm::Mat4 {
        let camera_translation = glm::translate(&glm::Mat4::identity(), &self.position);
        let camera_rotation = self.get_rotation_matrix();
        glm::inverse(&(camera_translation * camera_rotation))
    }

    pub fn get_rotation_matrix(&self) -> glm::Mat4 {
        let pitch_rotation = glm::quat_angle_axis(self.pitch, &glm::vec3(1.0, 0.0, 0.0));
        let yaw_rotation = glm::quat_angle_axis(self.yaw, &glm::vec3(0.0, -1.0, 0.0));

        glm::quat_to_mat4(&yaw_rotation) * glm::quat_to_mat4(&pitch_rotation)
    }
}
