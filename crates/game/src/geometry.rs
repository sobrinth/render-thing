use crate::physics::Aabb;
use engine::Vertex;
use nalgebra_glm as glm;

pub fn hex_aabb(center: glm::Vec3, radius: f32, height: f32) -> Aabb {
    let half_h = height * 0.5;
    let half_z = radius * (3.0_f32).sqrt() * 0.5;
    Aabb {
        min: glm::vec3(center.x - radius, center.y - half_h, center.z - half_z),
        max: glm::vec3(center.x + radius, center.y + half_h, center.z + half_z),
    }
}

pub fn hex_prism_mesh(radius: f32, height: f32) -> (Vec<Vertex>, Vec<u32>) {
    use std::f32::consts::PI;

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Flat-top rim: v_i at angle i*60° in XZ plane
    let rim: Vec<[f32; 2]> = (0..6)
        .map(|i| {
            let a = (i as f32) * PI / 3.0;
            [radius * a.cos(), radius * a.sin()]
        })
        .collect();

    let half_h = height * 0.5;

    // --- Top cap (vertices 0..=6, normal (0,1,0)) ---
    let top_base = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, half_h, 0.0],
        normal: [0.0, 1.0, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
        uv_x: 0.5,
        uv_y: 0.5,
    });
    for &[x, z] in &rim {
        vertices.push(Vertex {
            position: [x, half_h, z],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.5 + x / (2.0 * radius),
            uv_y: 0.5 + z / (2.0 * radius),
        });
    }
    // Fan: (center, v_i, v_{i+1}) — CW from above = correct for this renderer
    for i in 0..6u32 {
        indices.push(top_base);
        indices.push(top_base + 1 + i);
        indices.push(top_base + 1 + (i + 1) % 6);
    }

    // --- Bottom cap (vertices 7..=13, normal (0,-1,0)) ---
    let bot_base = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, -half_h, 0.0],
        normal: [0.0, -1.0, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
        uv_x: 0.5,
        uv_y: 0.5,
    });
    for &[x, z] in &rim {
        vertices.push(Vertex {
            position: [x, -half_h, z],
            normal: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_x: 0.5 + x / (2.0 * radius),
            uv_y: 0.5 + z / (2.0 * radius),
        });
    }
    // Reversed fan: (center, v_{i+1}, v_i)
    for i in 0..6u32 {
        indices.push(bot_base);
        indices.push(bot_base + 1 + (i + 1) % 6);
        indices.push(bot_base + 1 + i);
    }

    // --- 6 side quads (vertices 14..=37) ---
    for i in 0..6usize {
        let j = (i + 1) % 6;
        let [xi, zi] = rim[i];
        let [xj, zj] = rim[j];

        // Outward normal: perpendicular to edge at midpoint, at angle (30 + 60*i)°
        let a = (30.0 + 60.0 * i as f32).to_radians();
        let nx = a.cos();
        let nz = a.sin();

        let base = vertices.len() as u32;
        // BL, BR, TR, TL
        vertices.push(Vertex {
            position: [xi, -half_h, zi],
            normal: [nx, 0.0, nz],
            color: [1.0; 4],
            uv_x: 0.0,
            uv_y: 1.0,
        });
        vertices.push(Vertex {
            position: [xj, -half_h, zj],
            normal: [nx, 0.0, nz],
            color: [1.0; 4],
            uv_x: 1.0,
            uv_y: 1.0,
        });
        vertices.push(Vertex {
            position: [xj, half_h, zj],
            normal: [nx, 0.0, nz],
            color: [1.0; 4],
            uv_x: 1.0,
            uv_y: 0.0,
        });
        vertices.push(Vertex {
            position: [xi, half_h, zi],
            normal: [nx, 0.0, nz],
            color: [1.0; 4],
            uv_x: 0.0,
            uv_y: 0.0,
        });

        // (BL, BR, TR) and (BL, TR, TL)
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_prism_mesh_vertex_and_index_count() {
        let (verts, idx) = hex_prism_mesh(1.0, 1.0);
        assert_eq!(verts.len(), 38, "expected 38 vertices");
        assert_eq!(idx.len(), 72, "expected 72 indices");
    }

    #[test]
    fn hex_prism_mesh_indices_in_range() {
        let (verts, idx) = hex_prism_mesh(1.0, 1.0);
        let n = verts.len() as u32;
        for &i in &idx {
            assert!(i < n, "index {} out of range (len {})", i, n);
        }
    }

    #[test]
    fn hex_prism_mesh_top_center_normal_and_y() {
        let (verts, _) = hex_prism_mesh(2.0, 1.0);
        // top cap center is vertex 0
        assert!(
            (verts[0].position[1] - 0.5).abs() < 1e-5,
            "top center y = height/2"
        );
        assert!(
            (verts[0].normal[1] - 1.0).abs() < 1e-5,
            "top center normal y = 1"
        );
        assert!(verts[0].normal[0].abs() < 1e-5);
        assert!(verts[0].normal[2].abs() < 1e-5);
    }

    #[test]
    fn hex_prism_mesh_bottom_center_normal_and_y() {
        let (verts, _) = hex_prism_mesh(2.0, 1.0);
        // bottom cap center is vertex 7
        assert!(
            (verts[7].position[1] - (-0.5)).abs() < 1e-5,
            "bottom center y = -height/2"
        );
        assert!(
            (verts[7].normal[1] - (-1.0)).abs() < 1e-5,
            "bottom center normal y = -1"
        );
        assert!(verts[7].normal[0].abs() < 1e-5);
        assert!(verts[7].normal[2].abs() < 1e-5);
    }

    #[test]
    fn hex_prism_mesh_index_count_divisible_by_3() {
        let (_, idx) = hex_prism_mesh(1.0, 1.0);
        assert_eq!(idx.len() % 3, 0);
    }

    #[test]
    fn hex_aabb_at_origin() {
        let b = hex_aabb(glm::vec3(0.0, 0.0, 0.0), 1.0, 1.0);
        assert!((b.min.x - (-1.0)).abs() < 1e-5);
        assert!((b.max.x - 1.0).abs() < 1e-5);
        assert!((b.min.y - (-0.5)).abs() < 1e-5);
        assert!((b.max.y - 0.5).abs() < 1e-5);
        let expected_z = (3.0_f32).sqrt() * 0.5;
        assert!((b.min.z - (-expected_z)).abs() < 1e-5);
        assert!((b.max.z - expected_z).abs() < 1e-5);
    }

    #[test]
    fn hex_aabb_with_offset_center() {
        let b = hex_aabb(glm::vec3(10.0, 2.0, -5.0), 2.0, 1.0);
        assert!((b.min.x - 8.0).abs() < 1e-5);
        assert!((b.max.x - 12.0).abs() < 1e-5);
        assert!((b.min.y - 1.5).abs() < 1e-5);
        assert!((b.max.y - 2.5).abs() < 1e-5);
        let expected_half_z = 2.0 * (3.0_f32).sqrt() * 0.5;
        assert!((b.min.z - (-5.0 - expected_half_z)).abs() < 1e-5);
        assert!((b.max.z - (-5.0 + expected_half_z)).abs() < 1e-5);
    }
}
