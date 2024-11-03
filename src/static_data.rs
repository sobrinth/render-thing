use crate::Vertex;

pub const VERTEX_SIZE: usize = 20;
pub const VERTICES: [Vertex; 3] = [
    Vertex {
        pos: [0.0, -0.5],
        color: [1.0, 1.0, 1.0],
    },
    Vertex {
        pos: [0.5, 0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
];
