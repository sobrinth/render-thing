use crate::context::QueueData;
use crate::frame::FrameData;
use crate::meshes::MeshAsset;
use crate::pipeline::ComputeEffect;
use crate::renderer::FRAME_OVERLAP;
use crate::swapchain::SwapchainProperties;
use ash::Device;
use ash::vk::{CommandBuffer, Extent2D};
use egui::{ClippedPrimitive, Context, TextureId};
use egui_ash_renderer::Renderer;
use std::sync::Arc;
use vk_mem::Allocator;

pub(crate) struct UiContext {
    renderer: Option<Renderer>,
    pub ctx: Context,
    scale_factor: f32,
    textures_to_free: Option<Vec<TextureId>>,
}

impl UiContext {
    pub(crate) fn initialize(
        device: &Device,
        allocator: &Arc<Allocator>,
        swapchain_properties: SwapchainProperties,
    ) -> Self {
        let egui_renderer = Renderer::with_vk_mem_allocator(
            allocator.clone(),
            device.clone(),
            egui_ash_renderer::DynamicRendering {
                color_attachment_format: swapchain_properties.format.format,
                depth_attachment_format: None,
            },
            egui_ash_renderer::Options {
                in_flight_frames: FRAME_OVERLAP as usize,
                ..Default::default()
            },
        )
        .expect("Failed to create egui renderer");

        UiContext {
            renderer: Some(egui_renderer),
            ctx: Context::default(),
            scale_factor: 1.0,
            textures_to_free: None,
        }
    }
}

pub(crate) fn before_frame(
    ui: &mut UiContext,
    raw_input: egui::RawInput,
    graphics_queue: &QueueData,
    frame: &FrameData,
    mut active_data: (
        &mut ComputeEffect,
        &mut usize,
        &mut usize,
        &mut Option<Vec<MeshAsset>>,
        &mut f32,
        (u32, u32),
    ),
) -> (Vec<ClippedPrimitive>, egui::PlatformOutput) {
    let renderer = ui
        .renderer
        .as_mut()
        .expect("UI pre-draw call with renderer: 'None'");

    let ctx = ui.ctx.clone();

    ctx.begin_pass(raw_input);
    ui.scale_factor = ctx.pixels_per_point();
    egui::Window::new("Shader control")
        .resizable(false)
        .show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Render scale :");
                ui.add(egui::Slider::new(active_data.4, 0.1..=1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Effective resolution: ");
                ui.label(format!("{:.0}x{:.0}", active_data.5.0, active_data.5.1));
            });
            ui.add(egui::Separator::default().spacing(12.0));
            ui.horizontal(|ui| {
                ui.label("Selected background: ");
                ui.label(active_data.0.name);
            });
            ui.horizontal(|ui| {
                ui.label("Effect index:");
                ui.add(egui::Slider::new(active_data.1, 0..=2));
            });
            if let Some(meshes) = &mut active_data.3 {
                ui.horizontal(|ui| {
                    ui.label("Selected mesh: ");
                    ui.label(meshes[*active_data.2].name.as_str());
                });
                ui.horizontal(|ui| {
                    ui.label("Mesh index:");
                    ui.add(egui::Slider::new(active_data.2, 0..=meshes.len() - 1))
                });
            };
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

    let primitives = ctx.tessellate(shapes, pixels_per_point);

    if !textures_delta.free.is_empty() {
        ui.textures_to_free = Some(textures_delta.free.clone());
    }

    if !textures_delta.set.is_empty() {
        renderer
            .set_textures(
                graphics_queue.queue,
                frame.command_pool,
                textures_delta.set.as_slice(),
            )
            .unwrap();
    }

    (primitives, platform_output)
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
