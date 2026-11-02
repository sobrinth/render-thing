use nalgebra_glm as glm;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Bounds {
    pub origin: glm::Vec3,
    pub extents: glm::Vec3,
    pub sphere_radius: f32,
}
