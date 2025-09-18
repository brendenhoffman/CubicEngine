// SPDX-License-Identifier: CEPL-1.0
use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

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
}
