use crate::command_buffer::{CommandBuffer, Submitted, transition_image};
use crate::descriptor::{DescriptorWriter, GrowableAllocator};
use crate::pipeline::ComputePushConstants;
use crate::primitives::{GPUDrawPushConstants, GPUSceneData};
use crate::renderer::{FRAME_OVERLAP, VulkanRenderer};
use crate::resources::AllocatedBuffer;
use crate::sync::{Fence, Semaphore};
use crate::ui;
use ash::{Device, vk};
use glm::Mat4;

impl VulkanRenderer {
    pub(crate) fn draw(&mut self, raw_input: egui::RawInput) -> egui::PlatformOutput {
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
            (
                &mut res.background_effects[active_effect_idx],
                &mut res.active_background_effect,
                &mut res.active_mesh,
                &mut res.meshes,
                &mut res.render_scale,
                render_size,
            ),
        );

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

        self.draw_geometry(frame_index, cmd.handle(), draw_extent);

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

    fn draw_geometry(&mut self, frame_index: usize, cmd: vk::CommandBuffer, extent: vk::Extent2D) {
        // begin a render pass with the draw image
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.resources.draw_image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD) // maybe clear?
            .store_op(vk::AttachmentStoreOp::STORE);

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.resources.depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0f32,
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

        unsafe {
            self.context.device.cmd_begin_rendering(cmd, &render_info);
            self.context.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.resources.mesh_pipeline.pipeline,
            );
        }

        // Bind descriptor set for drawing
        let single_image_layout = self.resources.single_image_layout.layout;
        let image_set = self.resources.frames[frame_index]
            .descriptors
            .allocate(&self.context.device, single_image_layout);
        let mut descriptor_writer = DescriptorWriter::new();
        descriptor_writer.write_image(
            0,
            self.resources.checkerboard_image.view,
            self.resources.default_sampler_nearest.sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
        descriptor_writer.update_set(&self.context.device, image_set);

        unsafe {
            self.context.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.resources.mesh_pipeline.layout,
                0,
                &[image_set],
                &[],
            )
        }

        // dynamic viewport and scissor
        let viewport = vk::Viewport {
            x: 0f32,
            y: 0f32,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        unsafe { self.context.device.cmd_set_viewport(cmd, 0, &[viewport]) }

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        unsafe { self.context.device.cmd_set_scissor(cmd, 0, &[scissor]) }

        self.resources.main_camera.update();

        // Draw monkey head from meshes
        if let Some(meshes) = &self.resources.meshes {
            let mesh = &meshes[self.resources.active_mesh];

            // let view = glm::translate(&Mat4::identity(), &glm::vec3(0.0, 0.0, -5.0));
            let view = self.resources.main_camera.get_view_matrix();

            /*
               Use perspective_zo to get a projection matrix that is correct for vulkan
               C++ code often sets GLM_FORCE_DEPTH_ZERO_TO_ONE to make the normal glm::perspective work
               https://computergraphics.stackexchange.com/questions/12448/vulkan-perspective-matrix-vs-opengl-perspective-matrix

               This code uses 0 as the far plane and 1 as the near plane, so the values are switched (depth clear is set to 0.0f32)
            */
            let mut proj = glm::perspective_rh_zo(
                extent.width as f32 / extent.height as f32,
                70f32.to_radians(),
                10000f32,
                0.1f32,
            );

            let model = glm::scale(&Mat4::identity(), &glm::vec3(2.0, 2.0, 2.0));
            let model = glm::rotate_y(&model, 10f32.to_radians());

            proj[(1, 1)] *= -1.0; // Flip y axis due to GL <-> Vulkan difference in y-axis

            let push_constants = GPUDrawPushConstants {
                world_matrix: (proj * view * model).data.0,
                vertex_buffer: mesh.mesh_buffers.vertex_buffer_address,
            };

            unsafe {
                self.context.device.cmd_push_constants(
                    cmd,
                    self.resources.mesh_pipeline.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        (&push_constants as *const GPUDrawPushConstants).cast::<u8>(),
                        size_of::<GPUDrawPushConstants>(),
                    ),
                );
                self.context.device.cmd_bind_index_buffer(
                    cmd,
                    mesh.mesh_buffers.index_buffer.buffer,
                    0,
                    vk::IndexType::UINT32,
                );

                self.context.device.cmd_draw_indexed(
                    cmd,
                    mesh.surfaces[0].count,
                    1,
                    mesh.surfaces[0].start_index,
                    0,
                    0,
                );
            }
        }

        // dynamic temporal data
        let scene_mem_ptr = self.resources.frames[frame_index]
            .scene_buffer
            .info
            .mapped_data;

        unsafe {
            std::ptr::copy_nonoverlapping(
                (&self.resources.scene_data as *const GPUSceneData) as *const u8,
                scene_mem_ptr as *mut u8,
                size_of::<GPUSceneData>(),
            )
        }

        let scene_data_layout = self.resources.scene_data_layout.layout;
        let desc_set = self.resources.frames[frame_index]
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
        writer.update_set(&self.context.device, desc_set);

        unsafe {
            self.context.device.cmd_end_rendering(cmd);
        }
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
