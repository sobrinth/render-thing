use imgui::DrawData;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::time::Instant;
use winit::window::Window;

pub(crate) struct ImguiContext {
    _imgui: imgui::Context,
    _platform: WinitPlatform,
    _last_frame_time: Instant,
}

pub(crate) fn initialize(window: &'_ Window) -> ImguiContext {
    let mut imgui = imgui::Context::create();
    // configure imgui

    let mut platform = WinitPlatform::new(&mut imgui);
    platform.attach_window(imgui.io_mut(), window, HiDpiMode::Default);

    ImguiContext {
        _imgui: imgui,
        _platform: platform,
        _last_frame_time: Instant::now(),
    }
}

impl ImguiContext {
    pub(crate) fn draw_ui(&mut self, window: &Window) -> &DrawData {
        self._platform
            .prepare_frame(self._imgui.io_mut(), window)
            .expect("Failed to prepare frame");

        let ui = self._imgui.frame();
        // Do the rendering thingies

        self._platform.prepare_render(ui, window);
        self._imgui.render()
    }
}
