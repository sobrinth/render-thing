use engine::input::{ElementState, Key, MouseButton, NamedKey};
use nalgebra_glm as glm;

pub struct Camera {
    pub velocity: glm::Vec3,
    pub position: glm::Vec3,
    pitch: f32,
    yaw: f32,
    mouse_pressed: bool,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            velocity: glm::vec3(0.0, 0.0, 0.0),
            position: glm::vec3(0.0, 0.0, 5.0),
            pitch: 0.0,
            yaw: 0.0,
            mouse_pressed: false,
        }
    }

    pub fn handle_mouse_event(&mut self, old_pos: (i32, i32), new_pos: (i32, i32)) {
        let delta_x = new_pos.0 - old_pos.0;
        let delta_y = new_pos.1 - old_pos.1;
        if self.mouse_pressed {
            self.yaw += delta_x as f32 / 200f32;
            self.pitch -= delta_y as f32 / 200f32;
        }
    }

    pub fn handle_mouse_button_event(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.mouse_pressed = state == ElementState::Pressed;
        }
    }

    pub fn handle_key_event(&mut self, key_event: (ElementState, Key)) {
        match key_event.1 {
            Key::Character(c) => match key_event.0 {
                ElementState::Pressed => match c.as_str() {
                    "w" => self.velocity.z = -0.01f32,
                    "s" => self.velocity.z = 0.01f32,
                    "a" => self.velocity.x = -0.01f32,
                    "d" => self.velocity.x = 0.01f32,
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
                    NamedKey::Space => self.velocity.y = 0.01f32,
                    NamedKey::Shift => self.velocity.y = -0.01f32,
                    NamedKey::F2 | NamedKey::F3 | NamedKey::F4 | NamedKey::Other => {}
                },
                ElementState::Released => match n {
                    NamedKey::Space => self.velocity.y = 0f32,
                    NamedKey::Shift => self.velocity.y = 0f32,
                    NamedKey::F2 | NamedKey::F3 | NamedKey::F4 | NamedKey::Other => {}
                },
            },
            Key::Other => {}
        }
    }

    pub fn update(&mut self) {
        let rot = self.get_rotation_matrix();
        let movement = rot * glm::vec3_to_vec4(&self.velocity);
        self.position += glm::vec4_to_vec3(&movement);
    }

    pub fn get_view_matrix(&self) -> glm::Mat4 {
        let t = glm::translate(&glm::Mat4::identity(), &self.position);
        glm::inverse(&(t * self.get_rotation_matrix()))
    }

    pub fn get_rotation_matrix(&self) -> glm::Mat4 {
        let pitch = glm::quat_angle_axis(self.pitch, &glm::vec3(1.0, 0.0, 0.0));
        let yaw = glm::quat_angle_axis(self.yaw, &glm::vec3(0.0, -1.0, 0.0));
        glm::quat_to_mat4(&yaw) * glm::quat_to_mat4(&pitch)
    }
}
