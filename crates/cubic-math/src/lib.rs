// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
pub use glam::*;

/// A simple yaw/pitch fly camera. At yaw = pitch = 0 it looks down -Z with
/// +Y up, matching the renderer's right-handed convention.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    /// Radians; rotation around the world Y axis.
    pub yaw: f32,
    /// Radians; rotation around the camera's local X axis.
    pub pitch: f32,
    /// Vertical field of view, in radians.
    pub fovy: f32,
    /// Near clip distance. The far plane is infinite (reverse-Z).
    pub near: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fovy: std::f32::consts::FRAC_PI_3,
            near: 0.1,
        }
    }
}

impl Camera {
    /// Unit forward vector for the camera's current yaw/pitch.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            -self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }

    /// Right-handed view matrix looking from `position` along `forward()`.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    /// Right-handed, Vulkan depth range [0, 1], reverse-Z, infinite far
    /// plane projection matrix. Must stay numerically identical to what
    /// the renderer's pipeline expects (reverse-Z: clear depth = 0.0,
    /// depth_compare_op = GREATER_OR_EQUAL).
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        let f = 1.0 / (0.5 * self.fovy).tan();
        Mat4::from_cols(
            Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, self.near, 0.0),
        )
    }
}
