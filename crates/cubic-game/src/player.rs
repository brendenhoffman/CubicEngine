// SPDX-License-Identifier: CEPL-1.0

use crate::cubic::game::{camera, physics};

const PLAYER_HALF_W: f32 = 0.3; // 0.6m wide = 1.2 voxels
const PLAYER_HEIGHT: f32 = 1.8; // 1.8m tall = 3.6 voxels
const PLAYER_HALF_D: f32 = 0.3; // 0.6m deep = 1.2 voxels
const EYE_HEIGHT: f32 = 1.62; // eye level = 3.24 voxels
const GRAVITY: f32 = -20.0;
const JUMP_VELOCITY: f32 = 8.0;
const MOVE_SPEED: f32 = 4.5;
const THIRD_PERSON_DIST: f32 = 4.0;

/// Mirrors `cubic_wasm::InputSnapshot` field-for-field — see on_tick in
/// lib.rs, which decodes the host's get-input out-ptr buffer into this.
pub struct InputState {
    pub move_forward: bool,
    pub move_back: bool,
    pub move_left: bool,
    pub move_right: bool,
    pub jump: bool,
    // Not yet read anywhere: crouch isn't implemented (out of scope for
    // this card — see the module doc note in lib.rs), but the field stays
    // to keep this struct's layout matching cubic_wasm::InputSnapshot,
    // which the host's get-input out-ptr buffer is decoded against.
    #[allow(dead_code)]
    pub sneak: bool,
    pub look_dx: f32,
    pub look_dy: f32,
}

pub struct Player {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub grounded: bool,
    // Not yet reachable from any input binding — see the module doc note in
    // lib.rs. Left in place so the camera-placement math below is ready for
    // it once a toggle exists.
    pub third_person: bool,
}

impl Default for Player {
    fn default() -> Self {
        Self::new()
    }
}

impl Player {
    pub fn new() -> Self {
        Self {
            pos: [0.0, 20.0, 0.0],
            vel: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            grounded: false,
            third_person: false,
        }
    }

    pub fn eye_pos(&self) -> [f32; 3] {
        [self.pos[0], self.pos[1] + EYE_HEIGHT, self.pos[2]]
    }

    pub fn tick(&mut self, dt: f32, input: &InputState) {
        // 1. Gravity
        if !self.grounded {
            self.vel[1] += GRAVITY * dt;
        }
        self.vel[1] = self.vel[1].max(-50.0);

        // 2. Jump
        if input.jump && self.grounded {
            self.vel[1] = JUMP_VELOCITY;
            self.grounded = false;
        }

        // 3. Horizontal movement relative to yaw
        let sin_yaw = self.yaw.sin();
        let cos_yaw = self.yaw.cos();
        let forward = [-sin_yaw, 0.0, -cos_yaw];
        let right = [cos_yaw, 0.0, -sin_yaw];
        let mut mx = 0.0f32;
        let mut mz = 0.0f32;
        if input.move_forward {
            mx += forward[0];
            mz += forward[2];
        }
        if input.move_back {
            mx -= forward[0];
            mz -= forward[2];
        }
        if input.move_right {
            mx += right[0];
            mz += right[2];
        }
        if input.move_left {
            mx -= right[0];
            mz -= right[2];
        }
        let len = (mx * mx + mz * mz).sqrt();
        if len > 0.0 {
            mx = mx / len * MOVE_SPEED;
            mz = mz / len * MOVE_SPEED;
        }
        self.vel[0] = mx;
        self.vel[2] = mz;

        // 4. Mouse look
        self.yaw -= input.look_dx;
        self.pitch = (self.pitch - input.look_dy).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );

        // 5. Sweep through world
        let dx = self.vel[0] * dt;
        let dy = self.vel[1] * dt;
        let dz = self.vel[2] * dt;

        // Result buffer for sweep-aabb's out-ptr: [x, y, z, hit_x, hit_y,
        // hit_z] = 24 bytes. Stack-allocated is fine — the WASM stack lives
        // inside linear memory, so the host's mem.data_mut() write reaches
        // it just like any other address.
        let mut result_buf = [0u8; 24];
        let out_ptr = result_buf.as_mut_ptr() as u32;
        physics::sweep_aabb(
            self.pos[0],
            self.pos[1],
            self.pos[2],
            dx,
            dy,
            dz,
            PLAYER_HALF_W,
            PLAYER_HEIGHT,
            PLAYER_HALF_D,
            out_ptr,
        );
        // Parse result: [x:f32, y:f32, z:f32, hit_x:i32, hit_y:i32, hit_z:i32]
        let rx = f32::from_le_bytes(result_buf[0..4].try_into().unwrap());
        let ry = f32::from_le_bytes(result_buf[4..8].try_into().unwrap());
        let rz = f32::from_le_bytes(result_buf[8..12].try_into().unwrap());
        let hit_x = i32::from_le_bytes(result_buf[12..16].try_into().unwrap()) != 0;
        let hit_y = i32::from_le_bytes(result_buf[16..20].try_into().unwrap()) != 0;
        let hit_z = i32::from_le_bytes(result_buf[20..24].try_into().unwrap()) != 0;

        self.pos = [rx, ry, rz];

        // 6. Resolve velocity on collision
        if hit_x {
            self.vel[0] = 0.0;
        }
        if hit_y {
            if dy < 0.0 {
                self.grounded = true;
            }
            self.vel[1] = 0.0;
        } else {
            self.grounded = false;
        }
        if hit_z {
            self.vel[2] = 0.0;
        }

        // 7. Set camera
        let (cx, cy, cz, cyaw, cpitch) = if self.third_person {
            let eye = self.eye_pos();
            let bx = self.yaw.sin() * THIRD_PERSON_DIST;
            let bz = self.yaw.cos() * THIRD_PERSON_DIST;
            (
                eye[0] + bx,
                eye[1] + THIRD_PERSON_DIST * 0.3,
                eye[2] + bz,
                self.yaw,
                self.pitch - 0.2,
            )
        } else {
            let eye = self.eye_pos();
            (eye[0], eye[1], eye[2], self.yaw, self.pitch)
        };
        camera::set_camera(cx, cy, cz, cyaw, cpitch);
    }
}
