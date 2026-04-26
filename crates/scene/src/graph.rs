use engine::{DrawCall, MaterialHandle, MeshHandle};
use nalgebra_glm as glm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(pub u32);

pub struct SceneNode {
    pub name: String,
    pub local_transform: glm::Mat4,
    pub visible: bool,
    pub mesh: Option<MeshHandle>,
    pub material: Option<MaterialHandle>,
    pub(crate) children: Vec<NodeId>,
    pub(crate) parent: Option<NodeId>,
}

pub struct SceneGraph {
    nodes: Vec<SceneNode>,
    roots: Vec<NodeId>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
        }
    }

    pub fn add_root(&mut self, name: impl Into<String>, local_transform: glm::Mat4) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(SceneNode {
            name: name.into(),
            local_transform,
            visible: true,
            mesh: None,
            material: None,
            children: Vec::new(),
            parent: None,
        });
        self.roots.push(id);
        id
    }

    pub fn add_child(
        &mut self,
        parent: NodeId,
        name: impl Into<String>,
        local_transform: glm::Mat4,
    ) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(SceneNode {
            name: name.into(),
            local_transform,
            visible: true,
            mesh: None,
            material: None,
            children: Vec::new(),
            parent: Some(parent),
        });
        self.nodes[parent.0 as usize].children.push(id);
        id
    }

    pub fn node(&self, id: NodeId) -> &SceneNode {
        &self.nodes[id.0 as usize]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut SceneNode {
        &mut self.nodes[id.0 as usize]
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    pub fn children(&self, id: NodeId) -> &[NodeId] {
        &self.nodes[id.0 as usize].children
    }

    /// Moves all nodes from `other` into this graph, wiring `other`'s roots as children of `parent`.
    /// NodeIds previously issued by `other` must not be used with `self` after this call.
    pub fn adopt(&mut self, parent: NodeId, other: SceneGraph) {
        let offset = self.nodes.len() as u32;
        for mut node in other.nodes {
            node.children = node
                .children
                .into_iter()
                .map(|c| NodeId(c.0 + offset))
                .collect();
            node.parent = node.parent.map(|p| NodeId(p.0 + offset));
            self.nodes.push(node);
        }
        for root_id in other.roots {
            let adopted = NodeId(root_id.0 + offset);
            self.nodes[adopted.0 as usize].parent = Some(parent);
            self.nodes[parent.0 as usize].children.push(adopted);
        }
    }

    fn collect_draws(&self, id: NodeId, parent_world: &glm::Mat4, out: &mut Vec<DrawCall>) {
        let node = &self.nodes[id.0 as usize];
        if !node.visible {
            return;
        }
        let world = parent_world * node.local_transform;
        if let (Some(mesh), Some(material)) = (node.mesh, node.material) {
            out.push(DrawCall {
                mesh,
                material,
                transform: world,
            });
        }
        for &child in &node.children {
            self.collect_draws(child, &world, out);
        }
    }

    pub fn flatten_visible(&self) -> Vec<DrawCall> {
        let mut out = Vec::new();
        for &root in &self.roots {
            self.collect_draws(root, &glm::Mat4::identity(), &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_graph_has_no_roots() {
        let scene = SceneGraph::new();
        assert!(scene.roots().is_empty());
    }

    #[test]
    fn add_root_appears_in_roots() {
        let mut scene = SceneGraph::new();
        let id = scene.add_root("root", glm::Mat4::identity());
        assert_eq!(scene.roots(), &[id]);
    }

    #[test]
    fn node_name_is_set() {
        let mut scene = SceneGraph::new();
        let id = scene.add_root("my_node", glm::Mat4::identity());
        assert_eq!(scene.node(id).name, "my_node");
    }

    #[test]
    fn add_child_links_parent_and_child() {
        let mut scene = SceneGraph::new();
        let parent = scene.add_root("parent", glm::Mat4::identity());
        let child = scene.add_child(parent, "child", glm::Mat4::identity());
        assert_eq!(scene.children(parent), &[child]);
        assert_eq!(scene.node(child).parent, Some(parent));
    }

    #[test]
    fn child_does_not_appear_in_roots() {
        let mut scene = SceneGraph::new();
        let parent = scene.add_root("parent", glm::Mat4::identity());
        let child = scene.add_child(parent, "child", glm::Mat4::identity());
        assert!(!scene.roots().contains(&child));
    }

    #[test]
    fn node_mut_modifies_in_place() {
        let mut scene = SceneGraph::new();
        let id = scene.add_root("n", glm::Mat4::identity());
        scene.node_mut(id).visible = false;
        assert!(!scene.node(id).visible);
    }

    #[test]
    fn adopt_moves_roots_under_parent() {
        let mut base = SceneGraph::new();
        let parent = base.add_root("parent", glm::Mat4::identity());
        let mut other = SceneGraph::new();
        other.add_root("adopted", glm::Mat4::identity());
        base.adopt(parent, other);
        assert_eq!(base.children(parent).len(), 1);
        assert_eq!(base.node(base.children(parent)[0]).name, "adopted");
    }

    #[test]
    fn flatten_empty_graph_is_empty() {
        assert!(SceneGraph::new().flatten_visible().is_empty());
    }

    #[test]
    fn group_node_without_mesh_produces_no_draw() {
        let mut scene = SceneGraph::new();
        scene.add_root("group", glm::Mat4::identity());
        assert!(scene.flatten_visible().is_empty());
    }

    #[test]
    fn node_with_mesh_and_material_produces_draw() {
        let mut scene = SceneGraph::new();
        let id = scene.add_root("n", glm::Mat4::identity());
        scene.node_mut(id).mesh = Some(MeshHandle(0));
        scene.node_mut(id).material = Some(MaterialHandle(0));
        assert_eq!(scene.flatten_visible().len(), 1);
    }

    #[test]
    fn hidden_node_produces_no_draw() {
        let mut scene = SceneGraph::new();
        let id = scene.add_root("n", glm::Mat4::identity());
        scene.node_mut(id).mesh = Some(MeshHandle(0));
        scene.node_mut(id).material = Some(MaterialHandle(0));
        scene.node_mut(id).visible = false;
        assert!(scene.flatten_visible().is_empty());
    }

    #[test]
    fn hidden_parent_suppresses_visible_child() {
        let mut scene = SceneGraph::new();
        let parent = scene.add_root("parent", glm::Mat4::identity());
        let child = scene.add_child(parent, "child", glm::Mat4::identity());
        scene.node_mut(child).mesh = Some(MeshHandle(0));
        scene.node_mut(child).material = Some(MaterialHandle(0));
        scene.node_mut(parent).visible = false;
        assert!(scene.flatten_visible().is_empty());
    }

    #[test]
    fn unhiding_parent_restores_child_draw() {
        let mut scene = SceneGraph::new();
        let parent = scene.add_root("parent", glm::Mat4::identity());
        let child = scene.add_child(parent, "child", glm::Mat4::identity());
        scene.node_mut(child).mesh = Some(MeshHandle(0));
        scene.node_mut(child).material = Some(MaterialHandle(0));
        scene.node_mut(parent).visible = false;
        assert!(scene.flatten_visible().is_empty());
        scene.node_mut(parent).visible = true;
        assert_eq!(scene.flatten_visible().len(), 1);
    }

    #[test]
    fn world_transform_is_parent_times_local() {
        let mut scene = SceneGraph::new();
        let parent_t = glm::translate(&glm::Mat4::identity(), &glm::vec3(1.0, 0.0, 0.0));
        let child_t = glm::translate(&glm::Mat4::identity(), &glm::vec3(0.0, 2.0, 0.0));
        let parent = scene.add_root("parent", parent_t);
        let child = scene.add_child(parent, "child", child_t);
        scene.node_mut(child).mesh = Some(MeshHandle(0));
        scene.node_mut(child).material = Some(MaterialHandle(0));
        let draws = scene.flatten_visible();
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].transform, parent_t * child_t);
    }
}
