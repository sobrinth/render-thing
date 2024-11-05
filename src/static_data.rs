use crate::primitives::Vertex;

pub(crate) const VERTICES: [Vertex; 8] = [
    // First quad
    Vertex {
        pos: [-0.5, -0.5, 0.0],
        color: [1.0, 1.0, 1.0],
        coords: [0.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5, 0.0],
        color: [1.0, 1.0, 1.0],
        coords: [1.0, 0.0],
    },
    Vertex {
        pos: [0.5, 0.5, 0.0],
        color: [1.0, 1.0, 1.0],
        coords: [1.0, 1.0],
    },
    Vertex {
        pos: [-0.5, 0.5, 0.0],
        color: [1.0, 1.0, 1.0],
        coords: [0.0, 1.0],
    },
    // Second quad
    Vertex {
        pos: [-0.5, -0.5, -0.5],
        color: [1.0, 1.0, 1.0],
        coords: [0.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5, -0.5],
        color: [1.0, 1.0, 1.0],
        coords: [1.0, 0.0],
    },
    Vertex {
        pos: [0.5, 0.5, -0.5],
        color: [1.0, 1.0, 1.0],
        coords: [1.0, 1.0],
    },
    Vertex {
        pos: [-0.5, 0.5, -0.5],
        color: [1.0, 1.0, 1.0],
        coords: [0.0, 1.0],
    },
];

pub(crate) const INDICES: [u16; 12] = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4];
