use crate::physics::Aabb;
use engine::input::{Key, NamedKey};
use nalgebra_glm as glm;
use std::collections::HashSet;

const EYE_HEIGHT: f32 = 1.7;
pub const MOVE_SPEED: f32 = 5.0;
pub const JUMP_VELOCITY: f32 = 6.0;
const GRAVITY: f32 = -20.0;
const PLAYER_HALF_WIDTH: f32 = 0.4;
const PLAYER_HEIGHT: f32 = 1.8;

pub struct Player {
    pub position: glm::Vec3,
    pub velocity: glm::Vec3,
    pub pitch: f32,
    pub yaw: f32,
    pub on_ground: bool,
}

impl Player {
    pub fn new() -> Self {
        Self {
            position: glm::vec3(0.0, 2.0, 0.0),
            velocity: glm::zero(),
            pitch: 0.0,
            yaw: 0.0,
            on_ground: false,
        }
    }

    pub fn update(&mut self, dt: f32, held_keys: &HashSet<Key>) {
        // Gravity / ground
        if self.on_ground {
            self.velocity.y = 0.0;
        } else {
            self.velocity.y += GRAVITY * dt;
        }

        // Jump
        if self.on_ground && held_keys.contains(&Key::Named(NamedKey::Space)) {
            self.velocity.y = JUMP_VELOCITY;
            self.on_ground = false;
        }

        // Horizontal movement in yaw-space (no pitch tilt)
        let mut local = glm::vec3(0.0f32, 0.0, 0.0);
        if held_keys.contains(&Key::Character("w".to_string())) {
            local.z -= 1.0;
        }
        if held_keys.contains(&Key::Character("s".to_string())) {
            local.z += 1.0;
        }
        if held_keys.contains(&Key::Character("a".to_string())) {
            local.x -= 1.0;
        }
        if held_keys.contains(&Key::Character("d".to_string())) {
            local.x += 1.0;
        }

        if glm::length(&local) > 0.0 {
            local = glm::normalize(&local) * MOVE_SPEED;
        }

        // Rotate horizontal movement by yaw
        let yaw_rot =
            glm::quat_to_mat4(&glm::quat_angle_axis(self.yaw, &glm::vec3(0.0, -1.0, 0.0)));
        let world = yaw_rot * glm::vec4(local.x, 0.0, local.z, 0.0);
        self.velocity.x = world.x;
        self.velocity.z = world.z;

        self.position += self.velocity * dt;
    }

    pub fn resolve_collision(&mut self, boxes: &[Aabb]) {
        let (new_pos, on_ground) = crate::physics::resolve_position(
            self.position,
            PLAYER_HALF_WIDTH,
            PLAYER_HEIGHT,
            boxes,
        );
        self.position = new_pos;
        if on_ground && self.velocity.y < 0.0 {
            self.velocity.y = 0.0;
        }
        self.on_ground = on_ground;
    }

    pub fn eye_position(&self) -> glm::Vec3 {
        self.position + glm::vec3(0.0, EYE_HEIGHT, 0.0)
    }

    pub fn view_matrix(&self) -> glm::Mat4 {
        let pitch = glm::quat_angle_axis(self.pitch, &glm::vec3(1.0, 0.0, 0.0));
        let yaw = glm::quat_angle_axis(self.yaw, &glm::vec3(0.0, -1.0, 0.0));
        let rot = glm::quat_to_mat4(&yaw) * glm::quat_to_mat4(&pitch);
        let t = glm::translate(&glm::Mat4::identity(), &self.eye_position());
        glm::inverse(&(t * rot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_applied_when_airborne() {
        let mut player = Player::new();
        player.on_ground = false;
        player.update(1.0 / 60.0, &HashSet::new());
        assert!(player.velocity.y < 0.0);
    }

    #[test]
    fn no_gravity_when_on_ground() {
        let mut player = Player::new();
        player.on_ground = true;
        player.update(1.0 / 60.0, &HashSet::new());
        assert_eq!(player.velocity.y, 0.0);
    }

    #[test]
    fn jump_when_on_ground() {
        let mut player = Player::new();
        player.on_ground = true;
        let mut keys = HashSet::new();
        keys.insert(Key::Named(NamedKey::Space));
        player.update(1.0 / 60.0, &keys);
        assert!(player.velocity.y > 0.0);
        assert!(!player.on_ground);
    }

    #[test]
    fn no_jump_when_airborne() {
        let mut player = Player::new();
        player.on_ground = false;
        let mut keys = HashSet::new();
        keys.insert(Key::Named(NamedKey::Space));
        player.update(1.0 / 60.0, &keys);
        assert!(player.velocity.y < 0.0);
    }

    #[test]
    fn resolve_collision_sets_on_ground() {
        let mut player = Player::new();
        player.position = glm::vec3(0.0, -0.05, 0.0);
        let floor = Aabb {
            min: glm::vec3(-100.0, -1.0, -100.0),
            max: glm::vec3(100.0, 0.0, 100.0),
        };
        player.resolve_collision(&[floor]);
        assert!(player.on_ground);
        assert!(player.position.y >= 0.0);
    }
}
