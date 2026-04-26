use crate::context::QueueData;
use crate::frame::FrameData;
use crate::pipeline::ComputeEffect;
use crate::renderer::FRAME_OVERLAP;
use crate::stats::StatsHistory;
use crate::swapchain::SwapchainProperties;
use ash::Device;
use ash::vk::{CommandBuffer, Extent2D};
use egui::{ClippedPrimitive, Context, TextureId};
use egui_ash_renderer::Renderer;
use std::sync::Arc;
use vk_mem::Allocator;

pub(crate) struct UiState<'a> {
    pub(crate) effect: &'a mut ComputeEffect,
    pub(crate) effect_index: &'a mut usize,
    pub(crate) render_scale: &'a mut f32,
    pub(crate) effective_resolution: (u32, u32),
    pub(crate) stats: &'a StatsHistory,
    pub(crate) show_stats: &'a mut bool,
    pub(crate) show_controls: &'a mut bool,
    pub(crate) ambient_color: &'a mut [f32; 4],
    pub(crate) sunlight_direction: &'a mut [f32; 4],
    pub(crate) sunlight_color: &'a mut [f32; 4],
}

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
    state: UiState<'_>,
) -> (Vec<ClippedPrimitive>, egui::PlatformOutput) {
    let renderer = ui
        .renderer
        .as_mut()
        .expect("UI pre-draw call with renderer: 'None'");

    let ctx = ui.ctx.clone();

    ctx.begin_pass(raw_input);
    ui.scale_factor = ctx.pixels_per_point();
    if *state.show_controls {
        egui::Window::new("Shader control (F2)")
            .resizable(false)
            .open(state.show_controls)
            .show(&ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Render scale :");
                    ui.add(egui::Slider::new(state.render_scale, 0.1..=1.0));
                });
                ui.horizontal(|ui| {
                    ui.label("Effective resolution: ");
                    ui.label(format!(
                        "{:.0}x{:.0}",
                        state.effective_resolution.0, state.effective_resolution.1
                    ));
                });
                ui.add(egui::Separator::default().spacing(12.0));
                ui.horizontal(|ui| {
                    ui.label("Selected background: ");
                    ui.label(state.effect.name);
                });
                ui.horizontal(|ui| {
                    ui.label("Effect index:");
                    ui.add(egui::Slider::new(state.effect_index, 0..=2));
                });
                ui.add(egui::Separator::default().spacing(12.0));
                ui.heading("Lighting");
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Ambient color: ");
                    state.ambient_color.iter_mut().for_each(|v| {
                        ui.add(egui::DragValue::new(v).range(0.0..=f32::MAX).speed(0.005));
                    });
                });
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Sun direction: ");
                    state.sunlight_direction.iter_mut().take(3).for_each(|v| {
                        ui.add(egui::DragValue::new(v).range(-1.0..=1.0).speed(0.005));
                    });
                });
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Sun color: ");
                    state.sunlight_color.iter_mut().for_each(|v| {
                        ui.add(egui::DragValue::new(v).range(0.0..=f32::MAX).speed(0.01));
                    });
                });
                ui.add(egui::Separator::default().spacing(12.0));
                ui.heading("Push constants");
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Data1: ");
                    state.effect.data.data1.iter_mut().for_each(|v| {
                        ui.add(egui::DragValue::new(v).speed(0.005));
                    })
                });
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Data2: ");
                    state.effect.data.data2.iter_mut().for_each(|v| {
                        ui.add(egui::DragValue::new(v).speed(0.005));
                    })
                });
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Data3: ");
                    state.effect.data.data3.iter_mut().for_each(|v| {
                        ui.add(egui::DragValue::new(v).speed(0.005));
                    })
                });
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Data4: ");
                    state.effect.data.data4.iter_mut().for_each(|v| {
                        ui.add(egui::DragValue::new(v).speed(0.005));
                    })
                });
            });
    }

    if *state.show_stats {
        let cur = &state.stats.current;
        let avg = &state.stats.average;
        let fps_cur = if cur.frametime_ms > 0.0 {
            1000.0 / cur.frametime_ms
        } else {
            0.0
        };
        let fps_avg = if avg.frametime_ms > 0.0 {
            1000.0 / avg.frametime_ms
        } else {
            0.0
        };

        egui::Window::new("Debug (F4)")
            .resizable(false)
            .open(state.show_stats)
            .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
            .show(&ctx, |ui| {
                let mono = |s: String| egui::Label::new(egui::RichText::new(s).monospace());
                egui::Grid::new("stats_grid")
                    .num_columns(3)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("");
                        ui.label("Current");
                        ui.label("Avg (64f)");
                        ui.end_row();

                        ui.label("FPS");
                        ui.add(mono(format!("{:>6.1}", fps_cur)));
                        ui.add(mono(format!("{:>6.1}", fps_avg)));
                        ui.end_row();

                        ui.label("Frame");
                        ui.add(mono(format!("{:>6.2} ms", cur.frametime_ms)));
                        ui.add(mono(format!("{:>6.2} ms", avg.frametime_ms)));
                        ui.end_row();

                        ui.label("Draw");
                        ui.add(mono(format!("{:>6.2} ms", cur.draw_time_ms)));
                        ui.add(mono(format!("{:>6.2} ms", avg.draw_time_ms)));
                        ui.end_row();

                        ui.label("Update");
                        ui.add(mono(format!("{:>6.2} ms", cur.update_time_ms)));
                        ui.add(mono(format!("{:>6.2} ms", avg.update_time_ms)));
                        ui.end_row();

                        ui.label("Draw calls");
                        ui.add(mono(format!("{:>6}", cur.draw_call_count)));
                        ui.add(mono(format!("{:>6}", avg.draw_call_count)));
                        ui.end_row();

                        ui.label("Triangles");
                        ui.add(mono(format!("{:>6}", cur.triangle_count)));
                        ui.add(mono(format!("{:>6}", avg.triangle_count)));
                        ui.end_row();

                        ui.label("Opaque");
                        ui.add(mono(format!("{:>6}", cur.opaque_count)));
                        ui.label("—");
                        ui.end_row();

                        ui.label("Transparent");
                        ui.add(mono(format!("{:>6}", cur.transparent_count)));
                        ui.label("—");
                        ui.end_row();

                        ui.label("Culled");
                        ui.add(mono(format!("{:>6}", cur.culled_count)));
                        ui.label("—");
                        ui.end_row();
                    });
            });
    }

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
