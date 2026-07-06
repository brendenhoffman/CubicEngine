// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

// ---------------------------------------------------------------------------
// Shared mesh types — defined here so both the renderer backend (cubic-render-vk)
// and the world/meshing system (cubic-world) can depend on cubic-render
// without either knowing about the other.
// ---------------------------------------------------------------------------

/// Per-vertex data matching the GPU vertex input layout.
#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub uv: [f32; 2],
    pub normal: [f32; 3],
    /// Index into the bindless texture array (see `PushData::tex_index`).
    pub tex_index: u32,
}

/// Per-draw push-constant data: model matrix, tint colour, and bindless
/// texture index. Padded to 16-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct PushData {
    pub model: [[f32; 4]; 4],
    pub tint: [f32; 4],
    /// Index into the bindless texture array.
    pub tex_index: u32,
    pub _pad: [u32; 3],
}

/// Opaque handle to a mesh uploaded via the renderer's `upload_mesh` API.
/// The inner index is `pub` so renderer backends can construct and unwrap
/// handles; user code should treat the value as opaque.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshHandle(pub u32);

// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct RenderSize {
    pub width: u32,
    pub height: u32,
}

pub trait Renderer {
    fn new(
        window: &dyn HasWindowHandle,
        display: &dyn HasDisplayHandle,
        size: RenderSize,
    ) -> Result<Self>
    where
        Self: Sized;

    fn resize(&mut self, size: RenderSize) -> Result<()>;
    fn render(&mut self) -> Result<()>;
    fn set_clear_color(&mut self, rgba: [f32; 4]);
    fn set_vsync(&mut self, _on: bool) {}
    fn free_mesh(&mut self, _handle: MeshHandle) {} // default no-op
    fn upload_texture(&mut self, _pixels: &[u8], _width: u32, _height: u32) -> Result<u32> {
        Ok(0) // default no-op
    }
}
