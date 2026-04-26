use crate::graph::{NodeId, SceneGraph};
use egui::Context;

pub struct ScenePanel {
    selected: Option<NodeId>,
}

impl ScenePanel {
    pub fn new() -> Self {
        Self { selected: None }
    }

    pub fn show(&mut self, ctx: &Context, scene: &mut SceneGraph) {
        egui::Window::new("Scene").resizable(true).show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    let roots: Vec<NodeId> = scene.roots().to_vec();
                    for root in roots {
                        Self::show_node(ui, scene, root, &mut self.selected);
                    }
                });

            if let Some(selected) = self.selected {
                ui.separator();
                Self::show_inspector(ui, scene, selected);
            }
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
        ui.horizontal(|ui| {
            let mut visible = scene.node(id).visible;
            if ui.checkbox(&mut visible, "").changed() {
                scene.node_mut(id).visible = visible;
            }
            let name = scene.node(id).name.clone();
            if ui.selectable_label(*selected == Some(id), name).clicked() {
                *selected = Some(id);
            }
        });
        if !children.is_empty() {
            ui.indent(egui::Id::new(id.0), |ui| {
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
            });

        ui.label("Local transform:");
        let t = node.local_transform;
        for row in 0..4usize {
            ui.monospace(format!(
                "[{:.2}  {:.2}  {:.2}  {:.2}]",
                t[(row, 0)],
                t[(row, 1)],
                t[(row, 2)],
                t[(row, 3)]
            ));
        }
    }
}
