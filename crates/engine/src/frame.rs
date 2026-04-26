use crate::command_buffer::{CommandBuffer, Submitted, transition_image};
use crate::descriptor::{DescriptorWriter, GrowableAllocator};
use crate::material::MaterialInstance;
use crate::meshes::Bounds;
use crate::pipeline::ComputePushConstants;
use crate::primitives::{GPUDrawPushConstants, GPUSceneData};
use crate::renderer::{FRAME_OVERLAP, VulkanRenderer};
use crate::resources::AllocatedBuffer;
use crate::stats::{DrawStats, FrameStats};
use crate::sync::{Fence, Semaphore};
use crate::ui::{self, UiState};
use ash::{Device, vk, vk::Handle};
use nalgebra_glm as glm;
use std::sync::Arc;
use std::time::Instant;

#[derive(Default)]
pub(crate) struct DrawContext {
    pub opaque_surfaces: Vec<RenderObject>,
    pub transparent_surfaces: Vec<RenderObject>,
    pub culled_count: u32,
}

pub(crate) struct RenderObject {
    pub index_count: u32,
    pub first_index: u32,
    pub index_buffer: vk::Buffer,
    pub material: Arc<MaterialInstance>,
    pub transform: glm::Mat4,
    pub vertex_buffer_address: vk::DeviceAddress,
    pub bounds: Bounds,
}

/// Returns false if all 8 AABB corners project outside the same frustum half-space.
pub(crate) fn is_visible(obj: &RenderObject, view_proj: &glm::Mat4) -> bool {
    let mvp = view_proj * obj.transform;
    let o = obj.bounds.origin;
    let e = obj.bounds.extents;

    let corners = [
        glm::vec4(o.x + e.x, o.y + e.y, o.z + e.z, 1.0),
        glm::vec4(o.x + e.x, o.y + e.y, o.z - e.z, 1.0),
        glm::vec4(o.x + e.x, o.y - e.y, o.z + e.z, 1.0),
        glm::vec4(o.x + e.x, o.y - e.y, o.z - e.z, 1.0),
        glm::vec4(o.x - e.x, o.y + e.y, o.z + e.z, 1.0),
        glm::vec4(o.x - e.x, o.y + e.y, o.z - e.z, 1.0),
        glm::vec4(o.x - e.x, o.y - e.y, o.z + e.z, 1.0),
        glm::vec4(o.x - e.x, o.y - e.y, o.z - e.z, 1.0),
    ];

    let clip: Vec<glm::Vec4> = corners.iter().map(|c| mvp * c).collect();

    let planes: [fn(&glm::Vec4) -> bool; 6] = [
        |c| c.x > c.w,
        |c| c.x < -c.w,
        |c| c.y > c.w,
        |c| c.y < -c.w,
        |c| c.z > c.w,
        |c| c.z < 0.0,
    ];

    for outside in planes {
        if clip.iter().all(outside) {
            return false;
        }
    }
    true
}

impl VulkanRenderer {
    pub(crate) fn draw(
        &mut self,
        camera: crate::CameraView,
        draws: &[crate::DrawCall],
        raw_input: egui::RawInput,
    ) -> egui::PlatformOutput {
        let frame_start = Instant::now();
        let frametime_ms = self
            .resources
            .last_frame_start
            .replace(frame_start)
            .map(|prev| prev.elapsed().as_secs_f32() * 1000.0)
            .unwrap_or(0.0);
        if self.resources.resize_requested {
            let minimized = Self::resize_swapchain(self);
            if minimized {
                return egui::PlatformOutput::default();
            }
        }

        let draw_extent = (
            f32::min(
                self.resources.swapchain.properties.extent.width as f32,
                self.resources.draw_image.extent.width as f32,
            ) * self.resources.render_scale,
            f32::min(
                self.resources.swapchain.properties.extent.height as f32,
                self.resources.draw_image.extent.height as f32,
            ) * self.resources.render_scale,
        );
        self.resources.render_size = (draw_extent.0 as u32, draw_extent.1 as u32);

        let draw_extent = vk::Extent2D {
            width: self.resources.render_size.0,
            height: self.resources.render_size.1,
        };

        const ONE_SECOND: u64 = 1_000_000_000;
        let frame_index: usize = (self.resources.frame_number % FRAME_OVERLAP) as usize;

        let raw_cmd = self.resources.frames[frame_index].main_command_buffer;

        // wait for the GPU to have finished the last rendering of this frame.
        assert!(
            self.resources.frames[frame_index]
                .render_fence
                .wait(ONE_SECOND),
            "render fence timed out"
        );
        self.resources.frames[frame_index].render_fence.reset();
        self.resources.frames[frame_index].clean_resources(&self.context.device);

        let res = unsafe {
            self.resources.swapchain.swapchain_fn.acquire_next_image(
                self.resources.swapchain.swapchain,
                ONE_SECOND,
                self.resources.frames[frame_index]
                    .acquire_semaphore
                    .handle(),
                vk::Fence::null(),
            )
        };

        let image_index = match res {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.resources.resize_requested = true;
                return egui::PlatformOutput::default();
            }
            Err(err) => panic!("Failed to acquire next image. Cause: {err}"),
        };

        // BEFORE FRAME
        // Extract values needed to avoid simultaneous borrow conflicts through ManuallyDrop
        let active_effect_idx = self.resources.active_background_effect;
        let render_size = self.resources.render_size;
        let res = &mut *self.resources;
        let (ui_primitives, platform_output) = ui::before_frame(
            &mut res.ui_context,
            raw_input,
            &res.graphics_queue,
            &res.frames[frame_index],
            UiState {
                effect: &mut res.background_effects[active_effect_idx],
                effect_index: &mut res.active_background_effect,
                render_scale: &mut res.render_scale,
                effective_resolution: render_size,
                stats: &res.stats,
                show_stats: &mut res.show_stats,
                show_controls: &mut res.show_controls,
                ambient_color: &mut res.scene_data.ambient_color,
                sunlight_direction: &mut res.scene_data.sunlight_direction,
                sunlight_color: &mut res.scene_data.sunlight_color,
            },
        );

        let t0 = Instant::now();
        let draw_ctx = self.update_scene(&camera, draws);
        let update_time_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // Reset and begin command buffer for the frame
        // SAFETY: render_fence was waited on and reset above; no other wrapper exists for this handle.
        let cmd = unsafe { CommandBuffer::<Submitted>::wrap(raw_cmd) };
        let cmd = cmd.reset(&self.context.device);
        let cmd = cmd.begin(
            &self.context.device,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        );

        // transition the main draw image to Layout::GENERAL so we can draw into it.
        // we will overwrite the contents, so we don't care about the old layout
        transition_image(
            &self.context.device,
            cmd.handle(),
            self.resources.draw_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        self.draw_background(cmd.handle(), &self.context.device, draw_extent);

        transition_image(
            &self.context.device,
            cmd.handle(),
            self.resources.draw_image.image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        transition_image(
            &self.context.device,
            cmd.handle(),
            self.resources.depth_image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

        let t0 = Instant::now();
        let draw_stats = self.draw_geometry(
            frame_index,
            cmd.handle(),
            draw_extent,
            &draw_ctx,
            camera.position,
        );
        let draw_time_ms = t0.elapsed().as_secs_f32() * 1000.0;

        self.draw_dev_overlay(frame_index, cmd.handle(), draw_extent, &camera.view_matrix);

        // transition the draw image and the swapchain image into their correct transfer layouts.
        transition_image(
            &self.context.device,
            cmd.handle(),
            self.resources.draw_image.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        transition_image(
            &self.context.device,
            cmd.handle(),
            self.resources.swapchain.images[image_index],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        // copy the draw image to the swapchain image
        Self::copy_image_to_image(
            &self.context.device,
            cmd.handle(),
            self.resources.draw_image.image,
            self.resources.swapchain.images[image_index],
            draw_extent,
            self.resources.swapchain.properties.extent,
        );

        // DO THE UI RENDER

        transition_image(
            &self.context.device,
            cmd.handle(),
            self.resources.swapchain.images[image_index],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.resources.swapchain.image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .store_op(vk::AttachmentStoreOp::STORE);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.resources.swapchain.properties.extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment_info));

        unsafe {
            self.context
                .device
                .cmd_begin_rendering(cmd.handle(), &rendering_info)
        }

        let swapchain_extent = self.resources.swapchain.properties.extent;
        ui::render(
            &mut self.resources.ui_context,
            cmd.handle(),
            swapchain_extent,
            ui_primitives,
        );

        unsafe { self.context.device.cmd_end_rendering(cmd.handle()) }

        // set the swapchain image to Layout::PRESENT so we can present it
        transition_image(
            &self.context.device,
            cmd.handle(),
            self.resources.swapchain.images[image_index],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        // finalize command buffer
        let cmd = cmd.end(&self.context.device);

        // Prepare queue submission
        // we want to wait on the present_semaphore, as that is signaled when the swapchain is ready,
        // we will signal render_semaphore, to signal rendering has finished
        let cmd_info = &[vk::CommandBufferSubmitInfo::default()
            .command_buffer(cmd.handle())
            .device_mask(0)];

        let wait_info = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(
                self.resources.frames[frame_index]
                    .acquire_semaphore
                    .handle(),
            )
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .device_index(0)
            .value(1)];

        let signal_info = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(self.resources.swapchain.semaphores[image_index].handle())
            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
            .device_index(0)
            .value(1)];

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait_info)
            .signal_semaphore_infos(signal_info)
            .command_buffer_infos(cmd_info);

        // submit a command buffer to the queue and execute it.
        // render_fence will now block until the graphic commands finish execution
        unsafe {
            self.context.device.queue_submit2(
                self.resources.graphics_queue.queue,
                &[submit_info],
                self.resources.frames[frame_index].render_fence.handle(),
            )
        }
        .unwrap();
        cmd.into_submitted(); // type-level marker: buffer is now pending, no Vulkan call

        // Prepare presentation
        // this will put the image just rendered to into the visible window
        // wait on render_semaphore for that, as it's necessary that drawing commands have finished
        let image_indices = &[image_index as u32];
        let present_semaphore = self.resources.swapchain.semaphores[image_index].handle();
        let present_semaphores = [present_semaphore];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(core::slice::from_ref(&self.resources.swapchain.swapchain))
            .wait_semaphores(&present_semaphores)
            .image_indices(image_indices);

        // TODO db: Maybe use `VK_EXT_swapchain_maintenance1` to be able to use a fence here and "circumvent" the semaphore per image
        let res = unsafe {
            self.resources
                .swapchain
                .swapchain_fn
                .queue_present(self.resources.graphics_queue.queue, &present_info)
        };

        match res {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.resources.resize_requested = true;
            }
            Err(e) => panic!("Failed to present swapchain image: {:?}", e),
        }

        ui::after_frame(&mut self.resources.ui_context);

        self.resources.stats.push(FrameStats {
            frametime_ms,
            draw_time_ms,
            update_time_ms,
            draw_call_count: draw_stats.draw_call_count,
            triangle_count: draw_stats.triangle_count,
            culled_count: draw_stats.culled_count,
            opaque_count: draw_stats.opaque_count,
            transparent_count: draw_stats.transparent_count,
        });
        // increase the number of frames drawn
        self.resources.frame_number += 1;

        platform_output
    }

    fn draw_background(&self, cmd: vk::CommandBuffer, gpu: &Device, extent: vk::Extent2D) {
        let active_background =
            &self.resources.background_effects[self.resources.active_background_effect];

        // bind the gradient drawing compute pipeline
        unsafe {
            gpu.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                active_background.pipeline.pipeline,
            )
        }

        // bind the descriptor set containing the draw image for the compute pipeline
        unsafe {
            gpu.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.resources.effect_pipeline_layout.layout,
                0,
                &[self.resources.draw_image_descriptors],
                &[],
            )
        }

        unsafe {
            gpu.cmd_push_constants(
                cmd,
                self.resources.effect_pipeline_layout.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    (&active_background.data as *const ComputePushConstants).cast::<u8>(),
                    size_of::<ComputePushConstants>(),
                ),
            )
        }

        // execute the compute pipeline dispatch using a 16x16 workgroup size, so we divide by 16
        unsafe {
            gpu.cmd_dispatch(
                cmd,
                f64::ceil(extent.width as f64 / 16.0) as u32,
                f64::ceil(extent.height as f64 / 16.0) as u32,
                1,
            )
        }
    }

    fn update_scene(
        &mut self,
        camera: &crate::CameraView,
        draws: &[crate::DrawCall],
    ) -> DrawContext {
        self.resources.scene_data.view = camera.view_matrix.data.0;
        self.resources.scene_data.proj = camera.proj_matrix.data.0;
        self.resources.scene_data.view_proj = (camera.proj_matrix * camera.view_matrix).data.0;
        self.resources.scene_data.camera_pos =
            [camera.position.x, camera.position.y, camera.position.z, 1.0];

        let view_proj =
            glm::Mat4::from_column_slice(self.resources.scene_data.view_proj.as_flattened());

        let mut ctx = DrawContext::default();
        let submitted = draws.len() as u32;
        for draw_call in draws {
            let mesh = &self.resources.mesh_registry[draw_call.mesh.0 as usize];
            let mat = Arc::clone(
                &self.resources.material_registry[draw_call.material.0 as usize].instance,
            );

            let obj = RenderObject {
                index_count: mesh.index_count,
                first_index: 0,
                index_buffer: mesh.buffers.index_buffer.buffer,
                material: mat,
                transform: draw_call.transform,
                vertex_buffer_address: mesh.buffers.vertex_buffer_address,
                bounds: mesh.bounds,
            };

            if is_visible(&obj, &view_proj) {
                match obj.material.pass_type {
                    crate::material::MaterialPass::MainColor => ctx.opaque_surfaces.push(obj),
                    crate::material::MaterialPass::Transparent => {
                        ctx.transparent_surfaces.push(obj)
                    }
                }
            }
        }
        let visible = (ctx.opaque_surfaces.len() + ctx.transparent_surfaces.len()) as u32;
        ctx.culled_count = submitted.saturating_sub(visible);
        ctx
    }

    fn draw_geometry(
        &mut self,
        frame_index: usize,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        ctx: &DrawContext,
        cam_pos: glm::Vec3,
    ) -> DrawStats {
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.resources.draw_image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.resources.depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });

        let render_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(core::slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);

        unsafe { self.context.device.cmd_begin_rendering(cmd, &render_info) }

        let mut stats = DrawStats::default();

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            self.context.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.context.device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );
        }

        // Write scene data into per-frame UBO and allocate + update set 0
        let scene_mem_ptr = self.resources.frames[frame_index]
            .scene_buffer
            .info
            .mapped_data;
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&self.resources.scene_data as *const GPUSceneData).cast::<u8>(),
                scene_mem_ptr.cast::<u8>(),
                size_of::<GPUSceneData>(),
            )
        }
        let scene_data_layout = self.resources.scene_data_layout.layout;
        let scene_set = self.resources.frames[frame_index]
            .descriptors
            .allocate(&self.context.device, scene_data_layout);
        let mut writer = DescriptorWriter::new();
        writer.write_buffer(
            0,
            self.resources.frames[frame_index].scene_buffer.buffer,
            size_of::<GPUSceneData>() as u64,
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
        writer.update_set(&self.context.device, scene_set);

        // Sort opaque surfaces by material pointer then index buffer to minimise state changes
        let mut opaque_indices: Vec<usize> = (0..ctx.opaque_surfaces.len()).collect();
        opaque_indices.sort_unstable_by(|&a, &b| {
            let oa = &ctx.opaque_surfaces[a];
            let ob = &ctx.opaque_surfaces[b];
            let ma = Arc::as_ptr(&oa.material) as usize;
            let mb = Arc::as_ptr(&ob.material) as usize;
            ma.cmp(&mb)
                .then(oa.index_buffer.as_raw().cmp(&ob.index_buffer.as_raw()))
        });

        stats.opaque_count = opaque_indices.len() as u32;
        stats.culled_count = ctx.culled_count;

        // Opaque pass — full state tracking
        let mut last_pipeline = vk::Pipeline::null();
        let mut last_material: *const crate::material::MaterialInstance = std::ptr::null();
        let mut last_index_buffer = vk::Buffer::null();
        // All materials share compatible set 0/1 layouts (GltfMetallicRoughness)

        for &i in &opaque_indices {
            let obj = &ctx.opaque_surfaces[i];
            let pipeline = obj.material.pipeline.pipeline;
            let layout = obj.material.pipeline.layout;
            let mat_ptr = Arc::as_ptr(&obj.material);

            if pipeline != last_pipeline {
                unsafe {
                    self.context.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );
                    self.context.device.cmd_set_viewport(cmd, 0, &[viewport]);
                    self.context.device.cmd_set_scissor(
                        cmd,
                        0,
                        &[vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent,
                        }],
                    );
                }
                last_pipeline = pipeline;
            }

            if mat_ptr != last_material {
                unsafe {
                    self.context.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        layout,
                        0,
                        &[scene_set, obj.material.material_set],
                        &[],
                    );
                }
                last_material = mat_ptr;
            }

            if obj.index_buffer != last_index_buffer {
                unsafe {
                    self.context.device.cmd_bind_index_buffer(
                        cmd,
                        obj.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }
                last_index_buffer = obj.index_buffer;
            }

            unsafe {
                let push = GPUDrawPushConstants {
                    world_matrix: obj.transform.data.0,
                    vertex_buffer: obj.vertex_buffer_address,
                };
                self.context.device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        (&push as *const GPUDrawPushConstants).cast::<u8>(),
                        size_of::<GPUDrawPushConstants>(),
                    ),
                );
                self.context.device.cmd_draw_indexed(
                    cmd,
                    obj.index_count,
                    1,
                    obj.first_index,
                    0,
                    0,
                );
            }
            stats.draw_call_count += 1;
            stats.triangle_count += obj.index_count / 3;
        }

        // Transparent pass — naive back-to-front sort by squared distance from camera
        let mut transparent: Vec<&RenderObject> = ctx.transparent_surfaces.iter().collect();
        transparent.sort_by(|a, b| {
            let t = |m: &glm::Mat4| glm::Vec3::from([m[(0, 3)], m[(1, 3)], m[(2, 3)]]);
            let da = (t(&a.transform) - cam_pos).norm_squared();
            let db = (t(&b.transform) - cam_pos).norm_squared();
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        stats.transparent_count = transparent.len() as u32;

        last_pipeline = vk::Pipeline::null();
        for obj in transparent {
            let pipeline = obj.material.pipeline.pipeline;
            let layout = obj.material.pipeline.layout;
            if pipeline != last_pipeline {
                unsafe {
                    self.context.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );
                }
                last_pipeline = pipeline;
            }
            unsafe {
                self.context.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[scene_set, obj.material.material_set],
                    &[],
                );
                self.context.device.cmd_bind_index_buffer(
                    cmd,
                    obj.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                let push = GPUDrawPushConstants {
                    world_matrix: obj.transform.data.0,
                    vertex_buffer: obj.vertex_buffer_address,
                };
                self.context.device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        (&push as *const GPUDrawPushConstants).cast::<u8>(),
                        size_of::<GPUDrawPushConstants>(),
                    ),
                );
                self.context.device.cmd_draw_indexed(
                    cmd,
                    obj.index_count,
                    1,
                    obj.first_index,
                    0,
                    0,
                );
            }
            stats.draw_call_count += 1;
            stats.triangle_count += obj.index_count / 3;
        }

        unsafe { self.context.device.cmd_end_rendering(cmd) }
        stats
    }

    fn draw_dev_overlay(
        &mut self,
        frame_index: usize,
        cmd: vk::CommandBuffer,
        draw_extent: vk::Extent2D,
        camera_view: &glm::Mat4,
    ) {
        if !self.resources.show_dev_overlay {
            return;
        }

        const GIZMO_SIZE: u32 = 180;
        let gx = (draw_extent.width / 2).saturating_sub(GIZMO_SIZE / 2);
        let gy = (draw_extent.height / 2).saturating_sub(GIZMO_SIZE / 2);

        // Rotation-only view: strip the translation column so the gizmo doesn't move with position
        let mut view = *camera_view;
        view[(0, 3)] = 0.0;
        view[(1, 3)] = 0.0;
        view[(2, 3)] = 0.0;

        // Swapped near/far so that the axis pointing *toward* the camera gets higher
        // depth and wins the GREATER_OR_EQUAL test — matching standard gizmo behaviour.
        // Flip Y for Vulkan so +Y points up in the viewport.
        let mut proj = glm::ortho_rh_zo(-1.5f32, 1.5, -1.5, 1.5, 5.0, -5.0);
        proj[(1, 1)] *= -1.0;

        let gizmo_scene = GPUSceneData {
            view: view.data.0,
            proj: proj.data.0,
            view_proj: (proj * view).data.0,
            ambient_color: [1.0, 1.0, 1.0, 1.0],
            sunlight_direction: [0.0, 1.0, 0.0, 1.0],
            sunlight_color: [1.0, 1.0, 1.0, 0.0], // w=0 disables sun → pure flat ambient colour
            camera_pos: [0.0, 0.0, 0.0, 1.0],
        };

        // Write gizmo scene data into the second GPUSceneData slot of the per-frame scene buffer
        let scene_mem_ptr = self.resources.frames[frame_index]
            .scene_buffer
            .info
            .mapped_data;
        unsafe {
            let dst = (scene_mem_ptr as *mut u8).add(size_of::<GPUSceneData>());
            std::ptr::copy_nonoverlapping(
                (&gizmo_scene as *const GPUSceneData).cast::<u8>(),
                dst,
                size_of::<GPUSceneData>(),
            );
        }

        // Allocate a descriptor set pointing at the second slot
        let scene_data_layout = self.resources.scene_data_layout.layout;
        let scene_buffer = self.resources.frames[frame_index].scene_buffer.buffer;
        let gizmo_scene_set = self.resources.frames[frame_index]
            .descriptors
            .allocate(&self.context.device, scene_data_layout);
        let mut writer = DescriptorWriter::new();
        writer.write_buffer(
            0,
            scene_buffer,
            size_of::<GPUSceneData>() as u64,
            size_of::<GPUSceneData>() as u64,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
        writer.update_set(&self.context.device, gizmo_scene_set);

        let gizmo_rect = vk::Rect2D {
            offset: vk::Offset2D {
                x: gx as i32,
                y: gy as i32,
            },
            extent: vk::Extent2D {
                width: GIZMO_SIZE,
                height: GIZMO_SIZE,
            },
        };

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.resources.draw_image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.resources.depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);

        let render_info = vk::RenderingInfo::default()
            .render_area(gizmo_rect)
            .layer_count(1)
            .color_attachments(core::slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);

        unsafe { self.context.device.cmd_begin_rendering(cmd, &render_info) }

        // Clear depth in the gizmo region to 0.0 (reversed-depth "far") so gizmo always appears on top
        let clear_attachment = vk::ClearAttachment {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            color_attachment: 0,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
        };
        let clear_rect = vk::ClearRect {
            rect: gizmo_rect,
            base_array_layer: 0,
            layer_count: 1,
        };
        unsafe {
            self.context
                .device
                .cmd_clear_attachments(cmd, &[clear_attachment], &[clear_rect]);
        }

        let viewport = vk::Viewport {
            x: gx as f32,
            y: gy as f32,
            width: GIZMO_SIZE as f32,
            height: GIZMO_SIZE as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            self.context.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.context.device.cmd_set_scissor(cmd, 0, &[gizmo_rect]);
        }

        let t = 0.05f32; // arm half-thickness
        let ct = 0.15f32; // cap half-thickness (3× arm)
        let ot = 0.1f32; // origin cube half-size

        // Arms start at the cube's outer face; stubs end at the cube's inner face.
        // Nothing overlaps in 3D, so depth testing handles all ordering naturally
        // with no extra clears needed.
        let m = &self.resources.gizmo_materials;
        let draws: [(crate::MaterialHandle, glm::Vec3, glm::Vec3); 10] = [
            (m[6], glm::vec3(-ot, -ot, -ot), glm::vec3(ot, ot, ot)), // origin cube
            (m[0], glm::vec3(ot, -t, -t), glm::vec3(0.75, t, t)),    // X body
            (m[0], glm::vec3(0.75, -ct, -ct), glm::vec3(1.0, ct, ct)), // X cap
            (m[3], glm::vec3(-0.3, -t, -t), glm::vec3(-ot, t, t)),   // X neg stub
            (m[1], glm::vec3(-t, ot, -t), glm::vec3(t, 0.75, t)),    // Y body
            (m[1], glm::vec3(-ct, 0.75, -ct), glm::vec3(ct, 1.0, ct)), // Y cap
            (m[4], glm::vec3(-t, -0.3, -t), glm::vec3(t, -ot, t)),   // Y neg stub
            (m[2], glm::vec3(-t, -t, ot), glm::vec3(t, t, 0.75)),    // Z body
            (m[2], glm::vec3(-ct, -ct, 0.75), glm::vec3(ct, ct, 1.0)), // Z cap
            (m[5], glm::vec3(-t, -t, -0.3), glm::vec3(t, t, -ot)),   // Z neg stub
        ];

        let gizmo_mesh_idx = self.resources.gizmo_mesh.0 as usize;
        let index_count = self.resources.mesh_registry[gizmo_mesh_idx].index_count;
        let index_buffer = self.resources.mesh_registry[gizmo_mesh_idx]
            .buffers
            .index_buffer
            .buffer;
        let vertex_buffer_address = self.resources.mesh_registry[gizmo_mesh_idx]
            .buffers
            .vertex_buffer_address;

        let mut last_pipeline = vk::Pipeline::null();
        for &(mat_handle, arm_min, arm_max) in &draws {
            let mat_idx = mat_handle.0 as usize;
            let pipeline = self.resources.material_registry[mat_idx]
                .instance
                .pipeline
                .pipeline;
            let layout = self.resources.material_registry[mat_idx]
                .instance
                .pipeline
                .layout;
            let mat_set = self.resources.material_registry[mat_idx]
                .instance
                .material_set;

            if pipeline != last_pipeline {
                unsafe {
                    self.context.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );
                }
                last_pipeline = pipeline;
            }

            unsafe {
                self.context.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[gizmo_scene_set, mat_set],
                    &[],
                );
                self.context.device.cmd_bind_index_buffer(
                    cmd,
                    index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                let center = (arm_min + arm_max) * 0.5;
                let scale = arm_max - arm_min;
                let transform =
                    glm::scale(&glm::translate(&glm::Mat4::identity(), &center), &scale);
                let push = GPUDrawPushConstants {
                    world_matrix: transform.data.0,
                    vertex_buffer: vertex_buffer_address,
                };
                self.context.device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        (&push as *const GPUDrawPushConstants).cast::<u8>(),
                        size_of::<GPUDrawPushConstants>(),
                    ),
                );
                self.context
                    .device
                    .cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
            }
        }

        unsafe { self.context.device.cmd_end_rendering(cmd) }
    }

    fn copy_image_to_image(
        device: &Device,
        cmd: vk::CommandBuffer,
        src: vk::Image,
        dst: vk::Image,
        src_size: vk::Extent2D,
        dst_size: vk::Extent2D,
    ) {
        let blit_region = &[vk::ImageBlit2::default()
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: src_size.width as i32,
                    y: src_size.height as i32,
                    z: 1,
                },
            ])
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: dst_size.width as i32,
                    y: dst_size.height as i32,
                    z: 1,
                },
            ])
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(0),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(0),
            )];

        let blit_info = vk::BlitImageInfo2::default()
            .src_image(src)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(dst)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .filter(vk::Filter::LINEAR)
            .regions(blit_region);

        unsafe { device.cmd_blit_image2(cmd, &blit_info) }
    }
}

pub struct FrameData {
    pub command_pool: vk::CommandPool,
    pub(crate) main_command_buffer: vk::CommandBuffer,
    pub(crate) acquire_semaphore: Semaphore,
    pub(crate) render_fence: Fence,
    pub(crate) descriptors: GrowableAllocator,
    pub(crate) scene_buffer: AllocatedBuffer,
    device: Device,
}

impl Drop for FrameData {
    fn drop(&mut self) {
        self.descriptors.destroy_pools(&self.device);
        // scene_buffer drops automatically via its own Drop
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl FrameData {
    pub(crate) fn new(
        command_pool: vk::CommandPool,
        main_command_buffer: vk::CommandBuffer,
        acquire_semaphore: Semaphore,
        render_fence: Fence,
        descriptors: GrowableAllocator,
        scene_buffer: AllocatedBuffer,
        device: Device,
    ) -> Self {
        Self {
            command_pool,
            main_command_buffer,
            acquire_semaphore,
            render_fence,
            descriptors,
            scene_buffer,
            device,
        }
    }

    pub(crate) fn clean_resources(&mut self, device: &Device) {
        self.descriptors.clear_pools(device);
    }
}
