// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Per-frame recording and submission: the render() entry point plus the
//! compute-cull / draw / present pipeline it drives each frame.

use anyhow::{anyhow, Result};
use ash::vk;
use ash::Entry;
use cubic_render::RenderSize;

use crate::instance::recreate_surface;
#[cfg(debug_assertions)]
use crate::pipeline::{create_pipeline, PipelineConfig};
use crate::resources::{
    depth_aspect_mask, depth_attachment_layout, DrawCandidate, MAX_INDIRECT_DRAWS,
};
#[cfg(debug_assertions)]
use crate::DeferredDrop;
use crate::{
    is_device_lost, is_surface_lost, is_swapchain_out_of_date, semaphore_submit_info_signal,
    semaphore_submit_info_wait, stage_flags2_from_legacy, GpuResource, VkRenderer,
};

impl VkRenderer {
    #[inline]
    fn should_skip_for_backoff(&mut self) -> bool {
        if self.backoff_frames > 0 {
            self.backoff_frames -= 1;
            true
        } else {
            false
        }
    }

    /// Destroy every trashed resource whose retirement value the timeline
    /// semaphore has already reached. Non-blocking: queries the semaphore's
    /// current counter rather than waiting on it.
    pub(crate) fn drain_trash(&mut self) {
        if self.trash.is_empty() {
            return;
        }
        // On query failure, fall back to 0 (i.e. drain nothing this round)
        // rather than risk destroying a resource still in use.
        let signaled =
            unsafe { self.device.get_semaphore_counter_value(self.timeline) }.unwrap_or(0);

        let mut i = 0;
        while i < self.trash.len() {
            if self.trash[i].value > signaled {
                i += 1;
                continue;
            }
            let item = self.trash.swap_remove(i);
            match item.resource {
                GpuResource::Buffer { buffer, alloc } => unsafe {
                    self.device.destroy_buffer(buffer, None);
                    let _ = self
                        .allocator
                        .as_mut()
                        .expect("allocator missing")
                        .free(alloc);
                },
                GpuResource::Image { image, alloc } => unsafe {
                    self.device.destroy_image(image, None);
                    let _ = self
                        .allocator
                        .as_mut()
                        .expect("allocator missing")
                        .free(alloc);
                },
                GpuResource::ImageView(view) => unsafe {
                    self.device.destroy_image_view(view, None);
                },
                GpuResource::Pipeline(p) => unsafe {
                    self.device.destroy_pipeline(p, None);
                },
                GpuResource::PipelineLayout(l) => unsafe {
                    self.device.destroy_pipeline_layout(l, None);
                },
                GpuResource::MeshSlot {
                    first_vertex,
                    vertex_count,
                    first_index,
                    index_count,
                } => {
                    self.vert_alloc.free(first_vertex, vertex_count);
                    self.idx_alloc.free(first_index, index_count);
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    fn hot_reload_shaders_if_changed(&mut self) -> Result<()> {
        let Some(dev) = self.shader_dev.as_mut() else {
            return Ok(());
        };

        let vm = std::fs::metadata(&dev.vert_spv)
            .and_then(|m| m.modified())
            .ok();
        let fm = std::fs::metadata(&dev.frag_spv)
            .and_then(|m| m.modified())
            .ok();

        let vert_changed = vm.is_some() && vm.unwrap() > dev.vert_mtime;
        let frag_changed = fm.is_some() && fm.unwrap() > dev.frag_mtime;

        if !(vert_changed || frag_changed) {
            return Ok(());
        }

        tracing::info!("vk: .spv change detected → rebuilding pipeline");

        // Update mtimes first to avoid tight loops if rebuild fails.
        if let Some(t) = vm {
            dev.vert_mtime = t;
        }
        if let Some(t) = fm {
            dev.frag_mtime = t;
        }

        // Ensure no in-flight use of old pipeline while swapping.
        unsafe {
            self.device.device_wait_idle().ok();
        }

        // Rebuild using the same loader (reads from shader_dir(), i.e.
        // CUBIC_SHADER_DIR if set, else assets/shaders/)
        let (new_layout, new_pipeline) = create_pipeline(
            &self.device,
            self.pipeline_cache,
            &PipelineConfig {
                color_format: self.format,
                depth_format: self.depth_format,
                set_layout_camera: self.desc_set_layout_camera,
                set_layout_material: self.desc_set_layout_material,
                set_layout_indirect_graphics: self.desc_set_layout_indirect_graphics,
            },
        )?;

        self.trash.push(DeferredDrop {
            value: self.timeline_value,
            resource: GpuResource::Pipeline(self.pipeline),
        });
        self.trash.push(DeferredDrop {
            value: self.timeline_value,
            resource: GpuResource::PipelineLayout(self.pipeline_layout),
        });
        self.pipeline_layout = new_layout;
        self.pipeline = new_pipeline;

        // No re-record needed here: render() records each frame's command
        // buffer fresh against whatever self.pipeline currently is.
        Ok(())
    }

    #[inline]
    fn transition_to_color(&self, cmd: vk::CommandBuffer, image: vk::Image) {
        let subrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let pre_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            image,
            subresource_range: subrange,
            ..Default::default()
        };

        let dep_pre = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &pre_barrier,
            ..Default::default()
        };
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep_pre) };
    }

    #[inline]
    fn transition_depth_to_attachment(&self, cmd: vk::CommandBuffer, image: vk::Image) {
        let subrange = vk::ImageSubresourceRange {
            aspect_mask: depth_aspect_mask(self.depth_format),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let pre = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            dst_access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: depth_attachment_layout(self.depth_format),
            image,
            subresource_range: subrange,
            ..Default::default()
        };
        let dep = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &pre,
            ..Default::default()
        };
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };
    }

    #[inline]
    fn begin_rendering(&self, cmd: vk::CommandBuffer, image_view: vk::ImageView) {
        let color_att = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: self.clear,
            ..Default::default()
        };

        let depth_att = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: self.depth_view,
            image_layout: depth_attachment_layout(self.depth_format),
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
            ..Default::default()
        };

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };

        let rendering_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            render_area,
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: &color_att,
            p_depth_attachment: &depth_att,
            ..Default::default()
        };

        unsafe { self.device.cmd_begin_rendering(cmd, &rendering_info) };
    }

    /// Phase 1 of the GPU-driven draw: write candidates, dispatch indirect-cull
    /// compute, and leave the indirect/count buffers ready for the draw call.
    /// Must run OUTSIDE the render pass (before vkCmdBeginRendering).
    fn cull_compute_prepass(&self, cmd: vk::CommandBuffer, image_index: usize) {
        let candidate_count = self.pending_draws.len() as u32;

        // Write this frame's DrawCandidate array to the host-mapped buffer.
        if candidate_count > 0 {
            let ptr = self.candidate_ptrs[image_index] as *mut DrawCandidate;
            for (i, (handle, push)) in self.pending_draws.iter().enumerate() {
                let mesh = match self.meshes.get(handle.0 as usize) {
                    Some(m) => m,
                    None => continue,
                };
                unsafe {
                    std::ptr::write(
                        ptr.add(i),
                        DrawCandidate {
                            model: push.model,
                            tint: push.tint,
                            first_vertex: mesh.first_vertex as u32,
                            first_index: mesh.first_index,
                            index_count: mesh.index_count,
                            tex_index: push.tex_index,
                        },
                    );
                }
            }
        }

        // --- Compute dispatch: expand candidates → indirect commands ---
        // Zero the draw-count atomics before the compute shader writes them.
        // TRANSFER_DST ensures vkCmdFillBuffer completes before COMPUTE reads.
        let fill_to_compute = vk::MemoryBarrier2 {
            s_type: vk::StructureType::MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            dst_access_mask: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            ..Default::default()
        };
        let compute_to_indirect = vk::MemoryBarrier2 {
            s_type: vk::StructureType::MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_access_mask: vk::AccessFlags2::SHADER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::DRAW_INDIRECT
                | vk::PipelineStageFlags2::VERTEX_SHADER,
            dst_access_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ
                | vk::AccessFlags2::SHADER_READ,
            ..Default::default()
        };
        unsafe {
            self.device
                .cmd_fill_buffer(cmd, self.draw_count_bufs[image_index], 0, 4, 0);
            let dep = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                memory_barrier_count: 1,
                p_memory_barriers: &fill_to_compute,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier2(cmd, &dep);

            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.indirect_cull_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.indirect_cull_pipeline_layout,
                0,
                std::slice::from_ref(&self.indirect_compute_desc_sets[image_index]),
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.indirect_cull_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&candidate_count),
            );
            let groups = candidate_count.div_ceil(64).max(1);
            self.device.cmd_dispatch(cmd, groups, 1, 1);

            let dep2 = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                memory_barrier_count: 1,
                p_memory_barriers: &compute_to_indirect,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier2(cmd, &dep2);
        }
    }

    /// Phase 2: the actual indirect draw call. Must run INSIDE the render pass
    /// (between vkCmdBeginRendering and vkCmdEndRendering).
    fn record_indirect_draws(&self, cmd: vk::CommandBuffer, image_index: usize) -> Result<()> {
        if self.pipeline == vk::Pipeline::null() {
            return Err(anyhow!("pipeline is VK_NULL_HANDLE at record time"));
        }
        let vp = vk::Viewport {
            x: 0.0,
            y: self.extent.height as f32,
            width: self.extent.width as f32,
            height: -(self.extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let sc = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };
        let sets = [
            self.desc_sets[image_index],                   // set 0: camera
            self.material_desc_set,                        // set 1: bindless textures
            self.indirect_graphics_desc_sets[image_index], // set 2: candidates
        ];
        let offsets = [0_u64];
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device
                .cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            self.device
                .cmd_set_scissor(cmd, 0, std::slice::from_ref(&sc));
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &sets,
                &[],
            );
            // One shared vertex/index buffer pair for all meshes.
            self.device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.shared_vbuf),
                &offsets,
            );
            self.device
                .cmd_bind_index_buffer(cmd, self.shared_ibuf, 0, vk::IndexType::UINT32);
            // GPU populates the indirect buffer and count; CPU has no per-draw
            // involvement beyond writing the candidate array above.
            self.device.cmd_draw_indexed_indirect_count(
                cmd,
                self.indirect_bufs[image_index],
                0,
                self.draw_count_bufs[image_index],
                0,
                MAX_INDIRECT_DRAWS,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
        }
        Ok(())
    }

    #[inline]
    fn transition_to_present(&self, cmd: vk::CommandBuffer, image: vk::Image) {
        let subrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let post_barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::NONE,
            dst_access_mask: vk::AccessFlags2::empty(),
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            image,
            subresource_range: subrange,
            ..Default::default()
        };

        let dep_post = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &post_barrier,
            ..Default::default()
        };
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep_post) };
    }

    // Records draws queued via draw_mesh() into the given image's command
    // buffer. Called fresh every frame for the just-acquired image (see
    // render()) — safe to reset because acquire_next_image only returns an
    // image index once the GPU is done with its previous use.
    fn record_one_command(
        &mut self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        image_view: vk::ImageView,
        image_index: usize,
    ) -> Result<()> {
        // reset + begin
        unsafe {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?
        };
        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            ..Default::default()
        };
        unsafe { self.device.begin_command_buffer(cmd, &begin)? };

        // body
        // Phase 1: compute cull — MUST happen outside the render pass.
        self.cull_compute_prepass(cmd, image_index);
        self.transition_to_color(cmd, image);
        self.transition_depth_to_attachment(cmd, self.depth_image);
        self.begin_rendering(cmd, image_view);
        // Phase 2: indirect draw — inside the render pass.
        self.record_indirect_draws(cmd, image_index)?;
        // Egui overlay, if queued — still inside the render pass, on top of
        // the scene, before the image transitions to present.
        self.record_egui(cmd)?;
        unsafe { self.device.cmd_end_rendering(cmd) };
        self.transition_to_present(cmd, image);

        // end
        unsafe { self.device.end_command_buffer(cmd)? };
        Ok(())
    }

    // STRICT PER-FRAME ORDER:
    // 1) acquire_next_image (waits on acquire semaphore)
    // 2) record this frame's draws into the acquired image's command buffer
    //    (acquire_next_image only returns an image once the GPU is done
    //    with its previous use, so resetting its command buffer here is safe)
    // 3) queue_submit (signals render-finished for THIS image)
    // 4) queue_present (waits on render-finished)
    // Each swapchain image has its own FrameSync; do not cross-use semaphores.
    pub(crate) fn render_frame(&mut self) -> Result<()> {
        // Guard on pause
        if self.paused {
            return Ok(());
        }
        // Backoff check
        if self.should_skip_for_backoff() {
            return Ok(());
        }
        #[cfg(debug_assertions)]
        self.hot_reload_shaders_if_changed()?;

        // 1) Acquire
        let acq_sem = self.acq_slots[self.acq_index].sem;
        let acq_last_signal_value = self.acq_slots[self.acq_index].last_signal_value;
        if acq_last_signal_value > 0 {
            let wait_info = vk::SemaphoreWaitInfo {
                s_type: vk::StructureType::SEMAPHORE_WAIT_INFO,
                flags: vk::SemaphoreWaitFlags::empty(),
                semaphore_count: 1,
                p_semaphores: &self.timeline,
                p_values: &acq_last_signal_value,
                ..Default::default()
            };
            unsafe {
                self.device.wait_semaphores(&wait_info, u64::MAX)?;
            }
        }

        self.drain_trash();

        let (image_index, _) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                acq_sem,
                vk::Fence::null(),
            )
        } {
            Ok(pair) => pair,
            Err(e) if is_swapchain_out_of_date(e) => {
                self.backoff_frames = 2;
                let want = RenderSize {
                    width: self.extent.width,
                    height: self.extent.height,
                };
                let _ = self.recreate_swapchain(want);
                return Ok(());
            }
            Err(e) if is_surface_lost(e) => {
                self.backoff_frames = 2;
                let entry = Entry::linked();
                if recreate_surface(
                    &entry,
                    &self.instance,
                    &self.surface_loader,
                    &mut self.surface,
                    self.display_raw,
                    self.window_raw,
                )
                .is_ok()
                {
                    let want = RenderSize {
                        width: self.extent.width,
                        height: self.extent.height,
                    };
                    let _ = self.recreate_swapchain(want);
                } else {
                    self.paused = true;
                }
                return Ok(());
            }
            Err(e) if is_device_lost(e) => return Err(anyhow!("vk: device lost during acquire")),
            Err(e) => return Err(anyhow!("acquire_next_image: {e:?}")),
        };

        let img = image_index as usize;
        let render_finished = self.frames[img].render_finished;
        let cmd = self.cmd_bufs[img];
        let aspect = self.extent.width as f32 / self.extent.height as f32;
        self.update_camera_ubo_for_image(img, &self.camera, aspect)?;

        // Record this frame's draws (queued via draw_mesh()) into the
        // image we just acquired, then clear the queue for the next frame.
        self.record_one_command(cmd, self.images[img], self.image_views[img], img)?;
        self.pending_draws.clear();

        // 2) Submit (wait on acquire sem; signal render-finished; bump timeline)
        let next_value = self.timeline_value.wrapping_add(1);

        let stage_color = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let stage2_color = stage_flags2_from_legacy(stage_color);

        // Build the semaphore infos
        let wait_acquire = semaphore_submit_info_wait(acq_sem, 0, stage2_color);
        let signal_present = semaphore_submit_info_signal(render_finished, 0, stage2_color);
        let signal_timeline = semaphore_submit_info_signal(self.timeline, next_value, stage2_color);

        // IMPORTANT: store in locals so the pointers in SubmitInfo2 stay valid
        let waits = [wait_acquire];
        let signals = [signal_present, signal_timeline];

        let cmd_info = vk::CommandBufferSubmitInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
            command_buffer: cmd,
            device_mask: 0,
            ..Default::default()
        };

        let submit2 = vk::SubmitInfo2 {
            s_type: vk::StructureType::SUBMIT_INFO_2,
            wait_semaphore_info_count: waits.len() as u32,
            p_wait_semaphore_infos: waits.as_ptr(),
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cmd_info,
            signal_semaphore_info_count: signals.len() as u32,
            p_signal_semaphore_infos: signals.as_ptr(),
            ..Default::default()
        };

        // Submit with robust error handling
        let submit_res = unsafe {
            self.device.queue_submit2(
                self.queue,
                std::slice::from_ref(&submit2),
                vk::Fence::null(),
            )
        };

        match submit_res {
            Ok(()) => {
                self.timeline_value = next_value;
                self.acq_slots[self.acq_index].last_signal_value = next_value;
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                return Err(anyhow!("vk: device lost during submit"));
            }
            Err(e) => {
                return Err(anyhow!("queue_submit2: {e:?}"));
            }
        }

        // 3) Present (wait on render-finished)
        let present = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            wait_semaphore_count: 1,
            p_wait_semaphores: &render_finished,
            swapchain_count: 1,
            p_swapchains: &self.swapchain,
            p_image_indices: &image_index,
            ..Default::default()
        };

        match unsafe { self.swapchain_loader.queue_present(self.queue, &present) } {
            Ok(_) => {}
            Err(e) if is_swapchain_out_of_date(e) => {
                self.backoff_frames = 2;
                let want = RenderSize {
                    width: self.extent.width,
                    height: self.extent.height,
                };
                let _ = self.recreate_swapchain(want);
                return Ok(());
            }
            Err(e) if is_surface_lost(e) => {
                self.backoff_frames = 2;
                let entry = Entry::linked();
                if recreate_surface(
                    &entry,
                    &self.instance,
                    &self.surface_loader,
                    &mut self.surface,
                    self.display_raw,
                    self.window_raw,
                )
                .is_ok()
                {
                    let want = RenderSize {
                        width: self.extent.width,
                        height: self.extent.height,
                    };
                    let _ = self.recreate_swapchain(want);
                } else {
                    self.paused = true;
                }
                return Ok(());
            }
            Err(e) if is_device_lost(e) => return Err(anyhow!("vk: device lost during present")),
            Err(e) => return Err(anyhow!("queue_present: {e:?}")),
        }

        // Rotate acquire slot
        self.acq_index = (self.acq_index + 1) % self.acq_slots.len();

        Ok(())
    }
}
