/*
typedef struct VkGraphicsPipelineCreateInfo {
    VkStructureType                                  sType;                 Rust, not relevant
    const void*                                      pNext;                 VkPipelineRenderingCreateInfo (dynamic rendering)
    VkPipelineCreateFlags                            flags;
    uint32_t                                         stageCount;            2 (combined with pStages in rust)
    const VkPipelineShaderStageCreateInfo*           pStages;               vertex & fragment shader module
    const VkPipelineVertexInputStateCreateInfo*      pVertexInputState;     ignore for now, pass send data and index manually
    const VkPipelineInputAssemblyStateCreateInfo*    pInputAssemblyState;   Triangle topology
    const VkPipelineTessellationStateCreateInfo*     pTessellationState;    No tesselation for now -> None
    const VkPipelineViewportStateCreateInfo*         pViewportState;        Default -> viewport will be set dynamically
    const VkPipelineRasterizationStateCreateInfo*    pRasterizationState;   Rasterization settings (depth bias, backface culling, wireframe, etc.)
    const VkPipelineMultisampleStateCreateInfo*      pMultisampleState;     Multisampling -> Default for now
    const VkPipelineDepthStencilStateCreateInfo*     pDepthStencilState;    Depth testing & stencil configuration
    const VkPipelineColorBlendStateCreateInfo*       pColorBlendState;      Color blending & alpha blending
    const VkPipelineDynamicStateCreateInfo*          pDynamicState;         Dynamic state -> viewport & scissor
    VkPipelineLayout                                 layout;                same as the compute pipeline layout
    VkRenderPass                                     renderPass;            None -> Dynamic rendering
    uint32_t                                         subpass;               None -> Dynamic rendering
    VkPipeline                                       basePipelineHandle;    Ignore
    int32_t                                          basePipelineIndex;     Ignore
} VkGraphicsPipelineCreateInfo;
 */
use ash::vk;

pub(crate) struct PipelineBuilder<'a> {
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    render_info: vk::PipelineRenderingCreateInfo<'a>,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    rasterizer: vk::PipelineRasterizationStateCreateInfo<'a>,
    multisampling: vk::PipelineMultisampleStateCreateInfo<'a>,
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'a>,
    pipline_layout: vk::PipelineLayout,
}

impl<'a> PipelineBuilder<'a> {
    pub(crate) fn init() -> Self {
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default();
        let render_info = vk::PipelineRenderingCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default();
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default();
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default();
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let pipline_layout = vk::PipelineLayout::null();

        Self {
            color_blend_attachment,
            render_info,
            shader_stages: Vec::new(),
            input_assembly,
            rasterizer,
            multisampling,
            depth_stencil,
            pipline_layout,
        }
    }

    pub(crate) fn build(&mut self, device: ash::Device) -> vk::Pipeline {
        // Only a single viewport and scissor is supported
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .scissor_count(1)
            .viewport_count(1);

        let blend_attachment = &[self.color_blend_attachment];

        // Dummy color and alpha blending. Functionally 'no-blend' but writing to the color attachment.
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(blend_attachment);

        // default as it's unnecessary
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

        let states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(states);

        // build graphics pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut self.render_info)
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&self.depth_stencil)
            .layout(self.pipline_layout)
            .dynamic_state(&dynamic_info);

        unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0]
        }
    }
}
