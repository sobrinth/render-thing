use crate::physics::Aabb;
use engine::{MaterialHandle, MeshHandle};
use nalgebra_glm as glm;
use scene::SceneGraph;

pub use scene::NodeId;

pub struct Level {
    pub scene: SceneGraph,
    pub collision_boxes: Vec<Aabb>,
}

impl Level {
    pub fn new(scene: SceneGraph, collision_boxes: Vec<Aabb>) -> Self {
        Self {
            scene,
            collision_boxes,
        }
    }

    pub fn all_draws(&self) -> Vec<engine::DrawCall> {
        self.scene.flatten_visible()
    }
}

pub fn build_box(
    scene: &mut SceneGraph,
    parent: NodeId,
    name: &str,
    box_mesh: MeshHandle,
    material: MaterialHandle,
    min: glm::Vec3,
    max: glm::Vec3,
) -> (NodeId, Aabb) {
    let center = (min + max) * 0.5;
    let scale = max - min;
    let transform = glm::scale(&glm::translate(&glm::Mat4::identity(), &center), &scale);
    let node_id = scene.add_child(parent, name, transform);
    scene.node_mut(node_id).mesh = Some(box_mesh);
    scene.node_mut(node_id).material = Some(material);
    let aabb = Aabb { min, max };
    (node_id, aabb)
}
