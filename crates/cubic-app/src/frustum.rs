// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use cubic_math::{Mat4, Vec3};

pub struct Frustum {
    // 5 planes only — near, left, right, top, bottom.
    // No far plane: the projection is infinite reverse-Z so the far plane
    // is at infinity and can never cull anything.
    planes: [[f32; 4]; 5],
}

impl Frustum {
    pub fn from_view_proj(m: &Mat4) -> Self {
        let r = [m.row(0), m.row(1), m.row(2), m.row(3)];
        let planes = [
            (r[3] + r[0]).into(), // left
            (r[3] - r[0]).into(), // right
            (r[3] + r[1]).into(), // bottom
            (r[3] - r[1]).into(), // top
            (r[3] + r[2]).into(), // near
                                  // far plane omitted — infinite reverse-Z projection
        ];
        let planes = planes.map(|p: [f32; 4]| {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            [p[0] / len, p[1] / len, p[2] / len, p[3] / len]
        });
        Self { planes }
    }

    /// Returns true if the AABB (min..max) is potentially visible.
    /// False means definitely outside — safe to skip.
    pub fn contains_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Find the corner of the AABB most in the direction of the plane normal
            // (the "positive vertex"). If even that corner is outside, the whole
            // AABB is outside.
            let px = if plane[0] >= 0.0 { max.x } else { min.x };
            let py = if plane[1] >= 0.0 { max.y } else { min.y };
            let pz = if plane[2] >= 0.0 { max.z } else { min.z };
            if plane[0] * px + plane[1] * py + plane[2] * pz + plane[3] < 0.0 {
                return false;
            }
        }
        true
    }
}
