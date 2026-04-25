use crate::physics::Aabb;
use engine::{DrawCall, GltfScene, MaterialHandle, MeshHandle};
use nalgebra_glm as glm;

pub struct Level {
    pub draws: Vec<DrawCall>,
    pub collision_boxes: Vec<Aabb>,
}

impl Level {
    pub fn new(
        gltf_scene: Option<GltfScene>,
        gltf_collision: Vec<Aabb>,
        proc_draws: Vec<DrawCall>,
        proc_collision: Vec<Aabb>,
    ) -> Self {
        let gltf_draws = gltf_scene.map(|s| s.draws).unwrap_or_default();
        let collision_boxes = gltf_collision.into_iter().chain(proc_collision).collect();
        let draws = gltf_draws.into_iter().chain(proc_draws).collect();
        Self {
            draws,
            collision_boxes,
        }
    }

    pub fn all_draws(&self) -> &[DrawCall] {
        &self.draws
    }
}

/// Scales a unit cube to cover [min, max]. Returns the draw call and matching AABB.
pub fn build_box(
    box_mesh: MeshHandle,
    material: MaterialHandle,
    min: glm::Vec3,
    max: glm::Vec3,
) -> (DrawCall, Aabb) {
    let center = (min + max) * 0.5;
    let scale = max - min;
    let transform = glm::scale(&glm::translate(&glm::Mat4::identity(), &center), &scale);
    let draw = DrawCall {
        mesh: box_mesh,
        material,
        transform,
    };
    let aabb = Aabb { min, max };
    (draw, aabb)
}
