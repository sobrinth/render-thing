use crate::graph::{NodeId, SceneGraph};
use egui::Context;
use nalgebra_glm as glm;

pub struct ScenePanel {
    selected: Option<NodeId>,
    visible: bool,
}

impl ScenePanel {
    pub fn new() -> Self {
        Self {
            selected: None,
            visible: false,
        }
    }

    pub fn show(&mut self, ctx: &Context, scene: &mut SceneGraph) {
        if ctx.input(|i| i.key_pressed(egui::Key::F1)) {
            self.visible = !self.visible;
        }
        if !self.visible {
            return;
        }
        let max_size = ctx.content_rect().size() * 0.90;
        egui::Window::new("Scene")
            .title_bar(false)
            .resizable([true, true])
            .max_size(max_size)
            .show(ctx, |ui| {
                // Claim bottom space first so the scroll area gets what remains.
                if let Some(selected) = self.selected {
                    egui::TopBottomPanel::bottom("scene_inspector")
                        .frame(egui::Frame::NONE)
                        .show_inside(ui, |ui| {
                            ui.separator();
                            Self::show_inspector(ui, scene, selected);
                        });
                }
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let roots: Vec<NodeId> = scene.roots().to_vec();
                        for root in roots {
                            Self::show_node(ui, scene, root, &mut self.selected);
                        }
                    });
            });
    }

    pub fn selected(&self) -> Option<NodeId> {
        self.selected
    }

    fn show_node(
        ui: &mut egui::Ui,
        scene: &mut SceneGraph,
        id: NodeId,
        selected: &mut Option<NodeId>,
    ) {
        let children: Vec<NodeId> = scene.children(id).to_vec();
        let is_selected = *selected == Some(id);

        if children.is_empty() {
            ui.horizontal(|ui| {
                let mut visible = scene.node(id).visible;
                if ui.checkbox(&mut visible, "").changed() {
                    scene.node_mut(id).visible = visible;
                }
                let name = scene.node(id).name.clone();
                if ui
                    .add(egui::Button::selectable(is_selected, name))
                    .clicked()
                {
                    *selected = Some(id);
                }
            });
        } else {
            egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                egui::Id::new(id.0),
                true,
            )
            .show_header(ui, |ui| {
                let mut visible = scene.node(id).visible;
                if ui.checkbox(&mut visible, "").changed() {
                    scene.node_mut(id).visible = visible;
                }
                let name = scene.node(id).name.clone();
                if ui
                    .add(egui::Button::selectable(is_selected, name))
                    .clicked()
                {
                    *selected = Some(id);
                }
            })
            .body(|ui| {
                for child in children {
                    Self::show_node(ui, scene, child, selected);
                }
            });
        }
    }

    fn show_inspector(ui: &mut egui::Ui, scene: &SceneGraph, id: NodeId) {
        let node = scene.node(id);
        ui.strong("Inspector");
        egui::Grid::new("scene_inspector")
            .num_columns(2)
            .spacing([10.0, 4.0])
            .show(ui, |ui| {
                ui.label("Name");
                ui.label(&node.name);
                ui.end_row();

                ui.label("Visible");
                ui.label(if node.visible { "yes" } else { "no" });
                ui.end_row();

                if let Some(mesh) = node.mesh {
                    ui.label("Mesh");
                    ui.label(format!("MeshHandle({})", mesh.0));
                    ui.end_row();
                }

                if let Some(mat) = node.material {
                    ui.label("Material");
                    ui.label(format!("MaterialHandle({})", mat.0));
                    ui.end_row();
                }

                ui.label("Children");
                ui.label(format!("{}", scene.children(id).len()));
                ui.end_row();

                ui.strong("Local transform");
                ui.label("");
                ui.end_row();

                let (tr, rot_deg, sc) = decompose(&node.local_transform);

                ui.label("Translation");
                ui.monospace(format!("{:.2}  {:.2}  {:.2}", tr.x, tr.y, tr.z));
                ui.end_row();

                ui.label("Rotation");
                ui.monospace(format!(
                    "{:.1}°  {:.1}°  {:.1}°",
                    rot_deg.x, rot_deg.y, rot_deg.z
                ));
                ui.end_row();

                ui.label("Scale");
                ui.monospace(format!("{:.2}  {:.2}  {:.2}", sc.x, sc.y, sc.z));
                ui.end_row();
            });
    }
}

/// Decomposes a 4x4 transform matrix into translation, XYZ Euler angles (degrees), and scale.
fn decompose(t: &glm::Mat4) -> (glm::Vec3, glm::Vec3, glm::Vec3) {
    let translation = glm::vec3(t[(0, 3)], t[(1, 3)], t[(2, 3)]);

    let sx = glm::vec3(t[(0, 0)], t[(1, 0)], t[(2, 0)]).magnitude();
    let sy = glm::vec3(t[(0, 1)], t[(1, 1)], t[(2, 1)]).magnitude();
    let sz = glm::vec3(t[(0, 2)], t[(1, 2)], t[(2, 2)]).magnitude();
    let scale = glm::vec3(sx, sy, sz);

    if sx < 1e-6 || sy < 1e-6 || sz < 1e-6 {
        return (translation, glm::Vec3::zeros(), scale);
    }

    // Normalised rotation matrix elements (row, col)
    let r00 = t[(0, 0)] / sx;
    let r10 = t[(1, 0)] / sx;
    let r20 = t[(2, 0)] / sx;
    let r21 = t[(2, 1)] / sy;
    let r22 = t[(2, 2)] / sz;

    // XYZ intrinsic Euler angles: R = Rz(γ) · Ry(β) · Rx(α)
    let pitch_y = (-r20).clamp(-1.0, 1.0).asin();
    let (roll_x, yaw_z) = if pitch_y.cos().abs() > 1e-6 {
        (r21.atan2(r22), r10.atan2(r00))
    } else {
        // Gimbal lock — absorb yaw into roll
        let r01 = t[(0, 1)] / sy;
        let r02 = t[(0, 2)] / sz;
        if pitch_y > 0.0 {
            (r01.atan2(r02), 0.0_f32)
        } else {
            ((-r01).atan2(-r02), 0.0_f32)
        }
    };

    let rotation_deg = glm::vec3(
        roll_x.to_degrees(),
        pitch_y.to_degrees(),
        yaw_z.to_degrees(),
    );
    (translation, rotation_deg, scale)
}
