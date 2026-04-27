use crate::physics::Aabb;
use engine::{MaterialHandle, MeshHandle};
use nalgebra_glm as glm;
use scene::SceneGraph;
use std::f32::consts::PI;

pub use scene::NodeId;

struct OrbitState {
    node_id: NodeId,
    aabb_index: usize,
    orbit_radius: f32,
    y_offset: f32,
    angular_velocity: f32,
    mesh_radius: f32,
    mesh_height: f32,
    angle: f32,
}

pub struct Level {
    pub scene: SceneGraph,
    pub collision_boxes: Vec<Aabb>,
    orbiting: Vec<OrbitState>,
}

impl Level {
    pub fn new(scene: SceneGraph, collision_boxes: Vec<Aabb>) -> Self {
        Self {
            scene,
            collision_boxes,
            orbiting: Vec::new(),
        }
    }

    /// Register a platform node to orbit its parent. `period_secs` is one full revolution.
    /// `aabb_index` must be the index of the platform's entry in `collision_boxes`.
    pub fn add_orbit(
        &mut self,
        node_id: NodeId,
        aabb_index: usize,
        orbit_radius: f32,
        y_offset: f32,
        period_secs: f32,
        mesh_radius: f32,
        mesh_height: f32,
    ) {
        self.orbiting.push(OrbitState {
            node_id,
            aabb_index,
            orbit_radius,
            y_offset,
            angular_velocity: 2.0 * PI / period_secs,
            mesh_radius,
            mesh_height,
            angle: 0.0,
        });
    }

    pub fn tick(&mut self, dt: f32) {
        for orbit in &mut self.orbiting {
            orbit.angle = (orbit.angle + orbit.angular_velocity * dt) % (2.0 * PI);
            let local = glm::vec3(
                orbit.angle.sin() * orbit.orbit_radius,
                orbit.y_offset,
                -orbit.angle.cos() * orbit.orbit_radius,
            );
            self.scene.node_mut(orbit.node_id).local_transform =
                glm::translate(&glm::Mat4::identity(), &local);
            let t = self.scene.global_transform(orbit.node_id);
            let world = glm::vec3(t[(0, 3)], t[(1, 3)], t[(2, 3)]);
            self.collision_boxes[orbit.aabb_index] =
                crate::geometry::hex_aabb(world, orbit.mesh_radius, orbit.mesh_height);
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

pub fn build_hex_platform(
    scene: &mut SceneGraph,
    parent: NodeId,
    name: &str,
    mesh: MeshHandle,
    material: MaterialHandle,
    local_center: glm::Vec3,
) -> NodeId {
    let transform = glm::translate(&glm::Mat4::identity(), &local_center);
    let id = scene.add_child(parent, name, transform);
    scene.node_mut(id).mesh = Some(mesh);
    scene.node_mut(id).material = Some(material);
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{MaterialHandle, MeshHandle};
    use nalgebra_glm as glm;
    use scene::SceneGraph;

    #[test]
    fn build_hex_platform_creates_child_with_correct_state() {
        let mut scene = SceneGraph::new();
        let parent = scene.add_root("root", glm::Mat4::identity());
        let local = glm::vec3(1.0, 2.0, -3.0);

        let node = build_hex_platform(
            &mut scene,
            parent,
            "hex",
            MeshHandle(0),
            MaterialHandle(0),
            local,
        );

        // parent is pub(crate) in the scene crate — verify relationship via children()
        assert!(scene.children(parent).contains(&node));
        assert_eq!(scene.node(node).mesh, Some(MeshHandle(0)));
        assert_eq!(scene.node(node).material, Some(MaterialHandle(0)));

        let expected = glm::translate(&glm::Mat4::identity(), &local);
        assert_eq!(scene.node(node).local_transform, expected);
    }
}
