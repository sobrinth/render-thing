use nalgebra_glm as glm;

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: glm::Vec3,
    pub max: glm::Vec3,
}

/// Returns the new feet position and whether the player landed on top of a box.
pub fn resolve_position(
    mut feet: glm::Vec3,
    half_w: f32,
    height: f32,
    boxes: &[Aabb],
) -> (glm::Vec3, bool) {
    let mut on_ground = false;

    for box_ in boxes {
        let player = Aabb {
            min: feet + glm::vec3(-half_w, 0.0, -half_w),
            max: feet + glm::vec3(half_w, height, half_w),
        };

        let ox = f32::min(player.max.x, box_.max.x) - f32::max(player.min.x, box_.min.x);
        let oy = f32::min(player.max.y, box_.max.y) - f32::max(player.min.y, box_.min.y);
        let oz = f32::min(player.max.z, box_.max.z) - f32::max(player.min.z, box_.min.z);

        if ox <= 0.0 || oy <= 0.0 || oz <= 0.0 {
            continue;
        }

        // Push along the axis of minimum penetration.
        if ox <= oy && ox <= oz {
            let dir: f32 = if (player.min.x + player.max.x) > (box_.min.x + box_.max.x) {
                1.0
            } else {
                -1.0
            };
            feet.x += dir * ox;
        } else if oy <= ox && oy <= oz {
            let dir: f32 = if (player.min.y + player.max.y) > (box_.min.y + box_.max.y) {
                1.0
            } else {
                -1.0
            };
            feet.y += dir * oy;
            if dir > 0.0 {
                on_ground = true;
            }
        } else {
            let dir: f32 = if (player.min.z + player.max.z) > (box_.min.z + box_.max.z) {
                1.0
            } else {
                -1.0
            };
            feet.z += dir * oz;
        }
    }

    (feet, on_ground)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor() -> Aabb {
        Aabb {
            min: glm::vec3(-100.0, -1.0, -100.0),
            max: glm::vec3(100.0, 0.0, 100.0),
        }
    }

    #[test]
    fn no_overlap_unchanged() {
        let feet = glm::vec3(0.0, 5.0, 0.0);
        let (new_pos, on_ground) = resolve_position(feet, 0.4, 1.8, &[floor()]);
        assert!(!on_ground);
        assert!((new_pos.x - 0.0).abs() < 1e-5);
        assert!((new_pos.y - 5.0).abs() < 1e-5);
        assert!((new_pos.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn sinking_into_floor_pushed_up() {
        // feet slightly below floor surface (y=0)
        let feet = glm::vec3(0.0, -0.05, 0.0);
        let (new_pos, on_ground) = resolve_position(feet, 0.4, 1.8, &[floor()]);
        assert!(on_ground);
        assert!((new_pos.y - 0.0).abs() < 1e-4);
    }

    #[test]
    fn clipping_wall_pushed_sideways() {
        // player standing, slightly inside a wall at x=5
        let feet = glm::vec3(4.95, 0.5, 0.0);
        let wall = Aabb {
            min: glm::vec3(5.0, -10.0, -10.0),
            max: glm::vec3(6.0, 10.0, 10.0),
        };
        let (new_pos, on_ground) = resolve_position(feet, 0.4, 1.8, &[wall]);
        assert!(!on_ground);
        // pushed away from wall; player right edge must not overlap wall
        assert!(new_pos.x + 0.4 <= 5.0 + 1e-4);
    }
}
