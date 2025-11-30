use crate::renderer::{ComputeEffect, FRAME_OVERLAP, FrameData, QueueData};
use crate::swapchain::SwapchainProperties;
use ash::Device;
use ash::vk::{CommandBuffer, Extent2D};
use egui::{ClippedPrimitive, TextureId};
use egui_ash_renderer::Renderer;
use std::sync::Arc;
use vk_mem::Allocator;
use winit::window::Window;

pub(crate) struct UiContext {
    renderer: Option<Renderer>,
    pub state: Option<egui_winit::State>,
    scale_factor: f32,
    textures_to_free: Option<Vec<TextureId>>,
}

impl UiContext {
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
            Some(window.scale_factor() as f32 * 1.5),
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
                in_flight_frames: FRAME_OVERLAP as usize, // TODO: Frame overlap
                ..Default::default()
            },
        )
        .expect("Failed to create egui renderer");

        UiContext {
            renderer: Some(egui_renderer),
            state: Some(gui_state),
            scale_factor: window.scale_factor() as f32,
            textures_to_free: None,
        }
    }
}

pub(crate) fn before_frame(
    ui: &mut UiContext,
    window: &Window,
    graphics_queue: &QueueData,
    frame: &FrameData,
    active_data: (&mut ComputeEffect, &mut usize),
) -> Vec<ClippedPrimitive> {
    let gui_state = ui
        .state
        .as_mut()
        .expect("UI pre-draw call with gui_state: 'None");
    let renderer = ui
        .renderer
        .as_mut()
        .expect("UI pre-draw call with renderer: 'None'");

    let input = gui_state.take_egui_input(window);
    let ctx = gui_state.egui_ctx().clone();

    ctx.begin_pass(input);
    egui::Window::new("Shader control")
        .resizable(false)
        .show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Selected background: ");
                ui.label(active_data.0.name);
            });
            ui.horizontal(|ui| {
                ui.label("Effect index:");
                ui.add(egui::Slider::new(active_data.1, 0..=1).text(""));
            });
            ui.add(egui::Separator::default().spacing(12.0));
            ui.heading("Push constants");
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label("Data1: ");
                active_data.0.data.data1.iter_mut().for_each(|v| {
                    ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005));
                })
            });
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label("Data2: ");
                active_data.0.data.data2.iter_mut().for_each(|v| {
                    ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005));
                })
            });
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label("Data3: ");
                active_data.0.data.data3.iter_mut().for_each(|v| {
                    ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005));
                })
            });
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label("Data4: ");
                active_data.0.data.data4.iter_mut().for_each(|v| {
                    ui.add(egui::DragValue::new(v).range(0.0..=1.0).speed(0.005));
                })
            })
        });

    let egui::FullOutput {
        platform_output,
        shapes,
        textures_delta,
        pixels_per_point,
        ..
    } = ctx.end_pass();

    gui_state.handle_platform_output(window, platform_output);

    let primitives = ctx.tessellate(shapes, pixels_per_point);

    if !textures_delta.free.is_empty() {
        ui.textures_to_free = Some(textures_delta.free.clone());
    }

    // TODO: This is allocated here and not yet cleaned up correctly (the freeing below must be called
    // after the rendering is done it seems
    // Should the textures be on the frame? hmm
    if !textures_delta.set.is_empty() {
        renderer
            .set_textures(
                graphics_queue.queue,
                frame.command_pool,
                textures_delta.set.as_slice(),
            )
            .unwrap();
    }
    primitives
}

pub(crate) fn render(
    egui: &mut UiContext,
    cmd: CommandBuffer,
    extent: Extent2D,
    primitives: Vec<ClippedPrimitive>,
) {
    egui.renderer
        .as_mut()
        .expect("UI draw call with renderer: 'None'")
        .cmd_draw(cmd, extent, egui.scale_factor, &primitives)
        .unwrap();
}

pub(crate) fn after_frame(ui: &mut UiContext) {
    // ? soundness with multiple frames in flight
    // ? move to after frame
    if let Some(textures) = ui.textures_to_free.take() {
        log::trace!("Freeing {} textures from previous frame", textures.len());
        ui.renderer
            .as_mut()
            .unwrap()
            .free_textures(textures.as_slice())
            .unwrap();
    }
}
