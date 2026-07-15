use ash::vk;
use egui::TextureId;
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum RendererError {
    #[error("A Vulkan error occured: {0}")]
    Vulkan(#[from] vk::Result),

    #[error("A io error occured: {0}")]
    Io(#[from] std::io::Error),

    #[error("An error occured when initializing the renderer: {0}")]
    Init(String),

    #[error("Bad texture ID: {0:?}")]
    BadTexture(TextureId),
}
