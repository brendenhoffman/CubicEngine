// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use crate::cubic::game::{camera, physics};

const PLAYER_HALF_W: f32 = 0.3; // 0.6m wide = 1.2 voxels
const PLAYER_HEIGHT: f32 = 1.8; // 1.8m tall = 3.6 voxels
const PLAYER_HALF_D: f32 = 0.3; // 0.6m deep = 1.2 voxels
pub const EYE_HEIGHT: f32 = 1.62; // eye level = 3.24 voxels
const THIRD_PERSON_DIST: f32 = 4.0;
/// How far below the player to probe for standing contact (see the ground
/// probe in tick, step 6b) — small enough not to falsely ground someone who
/// just left the surface, large enough to register a resting player whose
/// zero-velocity sweep can't detect its own contact.
const GROUND_PROBE_DIST: f32 = 0.05;
/// Fallback spawn height for `Player::default()`, which nothing currently
/// calls — `on_load` in lib.rs always constructs via `Player::new` with a
/// terrain-aware height instead (see its call site).
const DEFAULT_SPAWN_Y: f32 = 20.0;
/// Max block-interaction distance, in metres (~12 voxels) — same ballpark
/// as vanilla Minecraft's reach.
const REACH: f32 = 6.0;
/// Ray march step size, in metres — well under VOXEL_SIZE so the march
/// can't skip over a thin voxel boundary and tunnel through a face.
const RAY_STEP: f32 = 0.05;

/// Mirrors `cubic_wasm::InputSnapshot` field-for-field — see on_tick in
/// lib.rs, which decodes the host's get-input out-ptr buffer into this.
pub struct InputState {
    pub move_forward: bool,
    pub move_back: bool,
    pub move_left: bool,
    pub move_right: bool,
    pub jump: bool,
    // Doubles as flying's descend control (see Player::tick) — crouch
    // itself isn't implemented while grounded (out of scope for this card).
    pub sneak: bool,
    pub look_dx: f32,
    pub look_dy: f32,
    /// cfg.player.* from cubic.toml (host-resolved, layered through
    /// game_overrides.toml / profile.toml) — see lib.rs's on_tick, which
    /// decodes these from the host's get-input out-ptr buffer. Distinct
    /// from cfg.camera.move_speed, which only drives the free-fly debug
    /// camera used when no game is loaded.
    pub walk_speed: f32,
    pub fly_speed: f32,
    pub jump_velocity: f32,
    pub gravity: f32,
    pub sprint_multiplier: f32,
}

pub struct Player {
    // f64: absolute world position, precise at any distance from the
    // origin (see the f64-world-coordinates card). `vel` stays f32 —
    // velocity magnitude is always small/bounded.
    pub pos: [f64; 3],
    pub vel: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub grounded: bool,
    // Toggled by a "toggle_third_person" InputEvent — see on_tick in lib.rs.
    pub third_person: bool,
    // Toggled by a "fly" InputEvent (double-tap Space by default) — see
    // on_tick in lib.rs.
    pub flying: bool,
    // Toggled by a "spectate" InputEvent (pause-menu button or its keybind
    // — see on_tick in lib.rs). Unlike flying, spectating detaches the
    // camera from the player entirely: `pos`/`vel`/`grounded` stay frozen
    // wherever the player physically was, and spectator_pos moves freely
    // with no collision at all (not even a de-penetration/ground probe).
    pub spectating: bool,
    pub spectator_pos: [f64; 3],
    // Set/cleared by a "sprint" InputEvent (double-tap forward by default,
    // registered by this game's game_overrides.toml rather than being a
    // built-in engine control — see on_tick in lib.rs). Unlike the toggles
    // above, sprint tracks kind==1 (Released) too: it's meant to last only
    // as long as the double-tapped key stays held, not flip on/off.
    pub sprinting: bool,
    // The block type placed by a "place_block" InputEvent — set by a
    // "pick_block" InputEvent (middle-click), initialized to whatever
    // on_load's caller passes as a sensible starting block. See on_tick in
    // lib.rs, which handles break/place/pick.
    pub selected_block: u32,
    // This tick's raycast hit, if the player is looking at a block within
    // REACH — recomputed once per tick (see raycast_target) and consumed
    // by on_tick in lib.rs both for break/place/pick and for drawing the
    // highlight mesh. None while spectating (tick_spectator never touches
    // this) or when nothing solid is in range.
    pub target: Option<RayHit>,
}

/// A raycast hit against solid voxel geometry: the targeted (solid) block,
/// and the empty voxel immediately before it along the ray — Minecraft's
/// usual "break the thing you're looking at, place against the face you're
/// looking at" pair. Both are min-corner world positions (matches how block
/// positions are computed from chunk-local voxel indices elsewhere).
#[derive(Clone, Copy)]
pub struct RayHit {
    pub block: [f64; 3],
    pub place: [f64; 3],
}

impl Default for Player {
    fn default() -> Self {
        Self::new(DEFAULT_SPAWN_Y, 0)
    }
}

impl Player {
    /// `spawn_y` should be the actual queried terrain height at the spawn
    /// column plus a little clearance (see `on_load`'s call site) — a fixed
    /// height can end up below the real surface on seeds with tall noise
    /// output, spawning the player embedded in the ground with no way to
    /// sweep back out (see `sweep_aabb`'s de-penetration step, which only
    /// ever helps once already ticking, not for the very first placement).
    /// `default_block` is the initially selected block for placing — see
    /// `on_load`'s call site, which passes its terrain fallback block.
    pub fn new(spawn_y: f32, default_block: u32) -> Self {
        Self {
            pos: [0.0, spawn_y as f64, 0.0],
            vel: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            grounded: false,
            third_person: false,
            flying: false,
            spectating: false,
            spectator_pos: [0.0; 3],
            sprinting: false,
            selected_block: default_block,
            target: None,
        }
    }

    pub fn eye_pos(&self) -> [f64; 3] {
        [self.pos[0], self.pos[1] + EYE_HEIGHT as f64, self.pos[2]]
    }

    /// March a ray from the eye along the look direction (yaw/pitch —
    /// full 3D, unlike movement's horizontal-only forward vector) up to
    /// REACH, sampling `physics::is-solid` every RAY_STEP. Always aims from
    /// the eye regardless of `third_person`, so aiming stays consistent
    /// with where the player is actually looking rather than the orbited
    /// third-person camera's angle.
    fn raycast_target(&self) -> Option<RayHit> {
        let origin = self.eye_pos();
        let dir = [
            -self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        ];

        let mut prev_voxel: Option<[i32; 3]> = None;
        let mut t = 0.0f32;
        let voxel_size = crate::VOXEL_SIZE as f64;
        while t < REACH {
            let p = [
                origin[0] + dir[0] as f64 * t as f64,
                origin[1] + dir[1] as f64 * t as f64,
                origin[2] + dir[2] as f64 * t as f64,
            ];
            if physics::is_solid(p[0], p[1], p[2]) {
                let voxel = [
                    (p[0] / voxel_size).floor() as i32,
                    (p[1] / voxel_size).floor() as i32,
                    (p[2] / voxel_size).floor() as i32,
                ];
                let block = [
                    voxel[0] as f64 * voxel_size,
                    voxel[1] as f64 * voxel_size,
                    voxel[2] as f64 * voxel_size,
                ];
                let place = prev_voxel
                    .map(|pv| {
                        [
                            pv[0] as f64 * voxel_size,
                            pv[1] as f64 * voxel_size,
                            pv[2] as f64 * voxel_size,
                        ]
                    })
                    .unwrap_or(block);
                return Some(RayHit { block, place });
            }
            prev_voxel = Some([
                (p[0] / voxel_size).floor() as i32,
                (p[1] / voxel_size).floor() as i32,
                (p[2] / voxel_size).floor() as i32,
            ]);
            t += RAY_STEP;
        }
        None
    }

    pub fn tick(&mut self, dt: f32, input: &InputState) {
        // Mouse look applies the same way regardless of mode, so it runs
        // once up front rather than being duplicated in both the spectator
        // and normal branches below. third_person itself is now toggled
        // directly by the "toggle_third_person" InputEvent handler in
        // on_tick (lib.rs) — no edge-detection needed here anymore, since
        // the host only forwards that event once per configured trigger
        // (tap/double-tap), not continuously while held.
        self.yaw -= input.look_dx;
        self.pitch = (self.pitch - input.look_dy).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );

        if self.spectating {
            self.tick_spectator(dt, input);
            return;
        }

        // Recomputed every tick from the just-updated look direction, ahead
        // of movement/physics below — on_tick in lib.rs reads this after
        // Player::tick returns, both to draw the highlight mesh and to
        // resolve this tick's break/place/pick events (see BlockAction).
        self.target = self.raycast_target();

        // 1. Vertical control: flying replaces gravity/ground-jump with
        // direct jump/sneak-driven ascend/descend (collision still applies
        // below via sweep_aabb either way — flying only skips gravity, not
        // the world). Toggled by a double-tap-jump InputEvent; see on_tick
        // in lib.rs, which also zeroes vel[1] on that same toggle so this
        // switch is a clean hover instead of carrying over jump/fall speed.
        if self.flying {
            let mut vy = 0.0f32;
            if input.jump {
                vy += input.fly_speed;
            }
            if input.sneak {
                vy -= input.fly_speed;
            }
            self.vel[1] = vy;
        } else {
            if !self.grounded {
                self.vel[1] += input.gravity * dt;
            }
            self.vel[1] = self.vel[1].max(-50.0);

            if input.jump && self.grounded {
                self.vel[1] = input.jump_velocity;
                self.grounded = false;
            }
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
        let mut horiz_speed = if self.flying {
            input.fly_speed
        } else {
            input.walk_speed
        };
        if self.sprinting {
            horiz_speed *= input.sprint_multiplier;
        }
        let len = (mx * mx + mz * mz).sqrt();
        if len > 0.0 {
            mx = mx / len * horiz_speed;
            mz = mz / len * horiz_speed;
        }
        self.vel[0] = mx;
        self.vel[2] = mz;

        // 5. Sweep through world
        let dx = self.vel[0] * dt;
        let dy = self.vel[1] * dt;
        let dz = self.vel[2] * dt;

        // Result buffer for sweep-aabb's out-ptr: [x, y, z (f64 each), hit_x,
        // hit_y, hit_z (i32 each)] = 3*8 + 3*4 = 36 bytes. Stack-allocated is
        // fine — the WASM stack lives inside linear memory, so the host's
        // mem.data_mut() write reaches it just like any other address.
        let mut result_buf = [0u8; 36];
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
        // Parse result: [x:f64, y:f64, z:f64, hit_x:i32, hit_y:i32, hit_z:i32]
        let rx = f64::from_le_bytes(result_buf[0..8].try_into().unwrap());
        let ry = f64::from_le_bytes(result_buf[8..16].try_into().unwrap());
        let rz = f64::from_le_bytes(result_buf[16..24].try_into().unwrap());
        let hit_x = i32::from_le_bytes(result_buf[24..28].try_into().unwrap()) != 0;
        let hit_y = i32::from_le_bytes(result_buf[28..32].try_into().unwrap()) != 0;
        let hit_z = i32::from_le_bytes(result_buf[32..36].try_into().unwrap()) != 0;

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

        // 6b. Ground probe: a resting player has dy == 0, and a zero-delta
        // sweep tests movement *into* solid ground, not "is something
        // already touching me" — sitting exactly at the surface boundary
        // therefore reports hit_y = false every tick with no downward
        // velocity, flickering `grounded` false right after it was set true
        // (position doesn't visibly change since the very next tick's
        // gravity nudge re-triggers hit_y=true and reverts it, but any
        // input read on a false tick — jump, or flying's ascend — is
        // silently dropped). Independently probe a hair below the resolved
        // position; unlike the real sweep this always has a small nonzero
        // downward delta, so it actually detects standing contact.
        if !self.flying && !self.grounded {
            let mut probe_buf = [0u8; 36];
            let probe_ptr = probe_buf.as_mut_ptr() as u32;
            physics::sweep_aabb(
                self.pos[0],
                self.pos[1],
                self.pos[2],
                0.0,
                -GROUND_PROBE_DIST,
                0.0,
                PLAYER_HALF_W,
                PLAYER_HEIGHT,
                PLAYER_HALF_D,
                probe_ptr,
            );
            // hit_y is the second of three i32 flags starting at byte 24
            // (after three f64 position fields) — see the result_buf parse
            // above for the full layout.
            let probe_hit_y = i32::from_le_bytes(probe_buf[28..32].try_into().unwrap()) != 0;
            if probe_hit_y {
                self.grounded = true;
            }
        }

        // 7. Set camera
        let (cx, cy, cz, cyaw, cpitch) = if self.third_person {
            let (pos, yaw, pitch) = self.third_person_camera();
            (pos[0], pos[1], pos[2], yaw, pitch)
        } else {
            let eye = self.eye_pos();
            (eye[0], eye[1], eye[2], self.yaw, self.pitch)
        };
        camera::set_camera(cx, cy, cz, cyaw, cpitch);

        // Feet position, tracked separately from the camera — third-person
        // orbit moves the camera away from the player, so diagnostics need
        // this to keep showing where the player actually is.
        camera::set_player_feet(self.pos[0], self.pos[1], self.pos[2]);
    }

    /// Spectator update: free camera movement (same yaw-relative WASD +
    /// jump/sneak-for-vertical scheme as flying, reusing fly_speed since
    /// spectating is conceptually "flying, but detached and unsolid") with
    /// no collision at all and no effect on the actual player — `pos`/`vel`/
    /// `grounded` are left exactly as they were, so un-spectating drops the
    /// player right back where they physically stood the whole time.
    fn tick_spectator(&mut self, dt: f32, input: &InputState) {
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
            mx = mx / len * input.fly_speed;
            mz = mz / len * input.fly_speed;
        }
        let mut my = 0.0f32;
        if input.jump {
            my += input.fly_speed;
        }
        if input.sneak {
            my -= input.fly_speed;
        }

        // No sweep_aabb call at all: spectating is explicitly unsolid, so
        // this is a plain kinematic integration, not even a de-penetration
        // check.
        self.spectator_pos[0] += (mx * dt) as f64;
        self.spectator_pos[1] += (my * dt) as f64;
        self.spectator_pos[2] += (mz * dt) as f64;

        camera::set_camera(
            self.spectator_pos[0],
            self.spectator_pos[1],
            self.spectator_pos[2],
            self.yaw,
            self.pitch,
        );
        // Report the spectator's own position here, not the frozen player
        // — while spectating, "where am I" (for diagnostics) means the free
        // camera, not the body left behind.
        camera::set_player_feet(
            self.spectator_pos[0],
            self.spectator_pos[1],
            self.spectator_pos[2],
        );
    }

    /// Orbit camera position for third-person view: sits behind the eye
    /// point on a sphere of radius THIRD_PERSON_DIST, opposite the look
    /// direction, so looking up/down swings the camera over/under the
    /// player instead of just sliding it up and re-angling (which is what
    /// the old fixed-height-offset version did).
    fn third_person_camera(&self) -> ([f64; 3], f32, f32) {
        let eye = self.eye_pos();

        let horiz = THIRD_PERSON_DIST * self.pitch.cos();
        let back_x = self.yaw.sin() * horiz;
        let back_y = -self.pitch.sin() * THIRD_PERSON_DIST;
        let back_z = self.yaw.cos() * horiz;

        let cam_pos = [
            eye[0] + back_x as f64,
            eye[1] + back_y as f64 + 0.5, // small upward bias so head is visible at flat pitch
            eye[2] + back_z as f64,
        ];

        // Camera looks toward eye — same yaw and pitch as player
        (cam_pos, self.yaw, self.pitch)
    }
}
