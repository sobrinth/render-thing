#[derive(Default, Clone, Copy)]
pub(crate) struct FrameStats {
    pub frametime_ms: f32,
    pub draw_time_ms: f32,
    pub update_time_ms: f32,
    pub draw_call_count: u32,
    pub triangle_count: u32,
    pub culled_count: u32,
    pub opaque_count: u32,
    pub transparent_count: u32,
}

#[derive(Default)]
pub(crate) struct DrawStats {
    pub draw_call_count: u32,
    pub triangle_count: u32,
    pub culled_count: u32,
    pub opaque_count: u32,
    pub transparent_count: u32,
}

const HISTORY_SIZE: usize = 64;

pub(crate) struct StatsHistory {
    history: [FrameStats; HISTORY_SIZE],
    head: usize,
    pub current: FrameStats,
    pub average: FrameStats,
}

impl Default for StatsHistory {
    fn default() -> Self {
        Self {
            history: [FrameStats::default(); HISTORY_SIZE],
            head: 0,
            current: FrameStats::default(),
            average: FrameStats::default(),
        }
    }
}

impl StatsHistory {
    pub(crate) fn push(&mut self, frame: FrameStats) {
        self.history[self.head % HISTORY_SIZE] = frame;
        self.head = self.head.wrapping_add(1);
        self.current = frame;
        self.average = self.compute_average();
    }

    fn compute_average(&self) -> FrameStats {
        let n = HISTORY_SIZE as f32;
        let mut ft = 0f32;
        let mut dt = 0f32;
        let mut ut = 0f32;
        let mut dc = 0u64;
        let mut tc = 0u64;
        for s in &self.history {
            ft += s.frametime_ms;
            dt += s.draw_time_ms;
            ut += s.update_time_ms;
            dc += s.draw_call_count as u64;
            tc += s.triangle_count as u64;
        }
        FrameStats {
            frametime_ms: ft / n,
            draw_time_ms: dt / n,
            update_time_ms: ut / n,
            draw_call_count: (dc as f64 / n as f64).round() as u32,
            triangle_count: (tc as f64 / n as f64).round() as u32,
            // not averaged — displayed as "—" in the debug overlay
            culled_count: 0,
            opaque_count: 0,
            transparent_count: 0,
        }
    }
}
