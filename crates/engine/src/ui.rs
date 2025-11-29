use crate::renderer::{FrameData, QueueData};
use crate::swapchain::{Swapchain, SwapchainProperties};
use ash::{Device, vk};
use egui_ash_renderer::Renderer;
use std::sync::{Arc, Mutex};
use vk_mem::Allocator;
use winit::window::Window;

pub(crate) struct EguiContext {
    renderer: Renderer,
    gui_state: egui_winit::State,
}

impl EguiContext {
    pub(crate) fn initialize(
        window: &Window,
        device: &Device,
        allocator: &Arc<Allocator>,
        swapchain_properties: SwapchainProperties,
    ) -> Self {
        let gui_context = egui::Context::default();
        gui_context.set_pixels_per_point(window.scale_factor() as f32);

        let viewport_id = gui_context.viewport_id();
        let gui_state = egui_winit::State::new(
            gui_context,
            viewport_id,
            &window,
            Some(window.scale_factor() as f32),
            Some(winit::window::Theme::Dark),
            None,
        );

        let egui_renderer = Renderer::with_vk_mem_allocator(
            allocator.clone(),
            device.clone(),
            egui_ash_renderer::DynamicRendering {
                color_attachment_format: swapchain_properties.format.format,
                depth_attachment_format: None,
            },
            egui_ash_renderer::Options {
                in_flight_frames: 3, // TODO: Frame overlap
                ..Default::default()
            },
        )
        .expect("Failed to create egui renderer");

        EguiContext {
            renderer: egui_renderer,
            gui_state,
        }
    }
}

pub(crate) fn before_frame(
    egui: &mut EguiContext,
    window: &Window,
    graphics_queue: &QueueData,
    frame: &FrameData,
) {
    let input = egui.gui_state.take_egui_input(window);
    let ctx = egui.gui_state.egui_ctx().clone();

    ctx.begin_pass(input);
    egui::Window::new("DEBUG").show(&ctx, |ui| ui.heading("Debug"));

    let egui::FullOutput {
        platform_output,
        shapes,
        textures_delta,
        pixels_per_point,
        ..
    } = ctx.end_pass();

    egui.gui_state
        .handle_platform_output(window, platform_output);

    let primitives = ctx.tessellate(shapes, pixels_per_point);

    // TODO: This is allocated here and not yet cleaned up correctly (the freeing below must be called
    // after the rendering is done it seems
    if !textures_delta.set.is_empty() {
        egui.renderer
            .set_textures(
                graphics_queue.queue,
                frame.command_pool,
                textures_delta.set.as_slice(),
            )
            .unwrap();
    }

    if !textures_delta.free.is_empty() {
        egui.renderer.free_textures(&textures_delta.free).unwrap();
    }
}