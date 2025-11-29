use std::error::Error;
use imgui::{DrawCmd, DrawCmdParams, DrawData};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::time::Instant;
use ash::vk;
use winit::window::Window;

pub(crate) struct ImguiContext {
    _imgui: imgui::Context,
    _platform: WinitPlatform,
    _last_frame_time: Instant,
}

pub(crate) fn initialize(window: &'_ Window) -> ImguiContext {
    let mut imgui = imgui::Context::create();
    // configure imgui
    imgui.fonts().build_rgba32_texture();

    let mut platform = WinitPlatform::new(&mut imgui);
    platform.attach_window(imgui.io_mut(), window, HiDpiMode::Default);

    ImguiContext {
        _imgui: imgui,
        _platform: platform,
        _last_frame_time: Instant::now(),
    }
}

pub fn draw_imgui(command_buffer: vk::CommandBuffer, gpu: &ash::Device, pipeline: &vk::Pipeline, draw_data: &DrawData) -> Result<(), Box<dyn Error>> {
    if draw_data.total_vtx_count == 0 {
        return Ok(());
    }

    unsafe {
        gpu.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            *pipeline,
        )
    };

    let framebuffer_width = draw_data.framebuffer_scale[0] * draw_data.display_size[0];
    let framebuffer_height = draw_data.framebuffer_scale[1] * draw_data.display_size[1];
    let viewports = [vk::Viewport {
        width: framebuffer_width,
        height: framebuffer_height,
        max_depth: 1.0,
        ..Default::default()
    }];

    unsafe { gpu.cmd_set_viewport(command_buffer, 0, &viewports) };

    // Ortho projection
    // let projection = orthographic_vk(
    //     0.0,
    //     draw_data.display_size[0],
    //     0.0,
    //     -draw_data.display_size[1],
    //     -1.0,
    //     1.0,
    // );
    // unsafe {
    //     let push = any_as_u8_slice(&projection);
    //     gpu.cmd_push_constants(
    //         command_buffer,
    //         self.pipeline_layout,
    //         vk::ShaderStageFlags::VERTEX,
    //         0,
    //         push,
    //     )
    // };
    //
    // unsafe {
    //     gpu
    //         .cmd_bind_index_buffer(command_buffer, mesh.indices, 0, vk::IndexType::UINT16)
    // };
    //
    // unsafe {
    //     gpu
    //         .cmd_bind_vertex_buffers(command_buffer, 0, &[mesh.vertices], &[0])
    // };

    let mut index_offset = 0;
    let mut vertex_offset = 0;
    let clip_offset = draw_data.display_pos;
    let clip_scale = draw_data.framebuffer_scale;
    for draw_list in draw_data.draw_lists() {
        for command in draw_list.commands() {
            match command {
                DrawCmd::Elements {
                    count,
                    cmd_params:
                        DrawCmdParams {
                            clip_rect,
                            texture_id,
                            vtx_offset,
                            idx_offset,
                        },
                } => {
                    unsafe {
                        let clip_x = (clip_rect[0] - clip_offset[0]) * clip_scale[0];
                        let clip_y = (clip_rect[1] - clip_offset[1]) * clip_scale[1];
                        let clip_w = (clip_rect[2] - clip_offset[0]) * clip_scale[0] - clip_x;
                        let clip_h = (clip_rect[3] - clip_offset[1]) * clip_scale[1] - clip_y;

                        let scissors = [vk::Rect2D {
                            offset: vk::Offset2D {
                                x: (clip_x as i32).max(0),
                                y: (clip_y as i32).max(0),
                            },
                            extent: vk::Extent2D {
                                width: clip_w as _,
                                height: clip_h as _,
                            },
                        }];
                        gpu.cmd_set_scissor(command_buffer, 0, &scissors);
                    }

                    unsafe {
                        gpu.cmd_draw_indexed(
                            command_buffer,
                            count as _,
                            1,
                            index_offset + idx_offset as u32,
                            vertex_offset + vtx_offset as i32,
                            0,
                        )
                    };
                }
                DrawCmd::ResetRenderState => {
                    log::trace!("Reset render state command not yet supported")
                }
                DrawCmd::RawCallback { .. } => {
                    log::trace!("Raw callback command not yet supported")
                }
            }
        }

        index_offset += draw_list.idx_buffer().len() as u32;
        vertex_offset += draw_list.vtx_buffer().len() as i32;
    }

    Ok(())
}

impl ImguiContext {
    pub(crate) fn draw_ui(&mut self, window: &Window) -> &DrawData {
        self._platform
            .prepare_frame(self._imgui.io_mut(), window)
            .expect("Failed to prepare frame");

        let ui = self._imgui.frame();
        // Do the rendering thingies
        ui.text("Hello, world!");

        self._platform.prepare_render(ui, window);
        self._imgui.render()
    }
}
