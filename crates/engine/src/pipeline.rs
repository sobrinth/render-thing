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
use ash::vk::Bool32;

pub(crate) struct PipelineBuilder<'a> {
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    render_info: vk::PipelineRenderingCreateInfo<'a>,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    rasterizer: vk::PipelineRasterizationStateCreateInfo<'a>,
    multisampling: vk::PipelineMultisampleStateCreateInfo<'a>,
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'a>,
    pipline_layout: vk::PipelineLayout,
    color_attachment_format: vk::Format,
}

const VK_FALSE: Bool32 = 0u32;

#[allow(dead_code)]
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
            color_attachment_format: vk::Format::UNDEFINED,
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

    pub(crate) fn set_shaders(
        &mut self,
        vertex_shader: vk::ShaderModule,
        fragment_shader: vk::ShaderModule,
    ) {
        self.shader_stages.clear();
        let vertex_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader);

        let fragment_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader);
        self.shader_stages.push(vertex_info);
        self.shader_stages.push(fragment_info);
    }

    pub(crate) fn set_input_topology(&mut self, topology: vk::PrimitiveTopology) {
        self.input_assembly.topology = topology;
        self.input_assembly.primitive_restart_enable = VK_FALSE;
    }

    pub(crate) fn set_polygon_mode(&mut self, mode: vk::PolygonMode) {
        self.rasterizer.polygon_mode = mode;
        self.rasterizer.line_width = 1.0;
    }

    pub(crate) fn set_cull_mode(&mut self, mode: vk::CullModeFlags, front_face: vk::FrontFace) {
        self.rasterizer.cull_mode = mode;
        self.rasterizer.front_face = front_face;
    }

    pub(crate) fn set_multisampling_none(&mut self) {
        self.multisampling.sample_shading_enable = VK_FALSE;
        // default to no MSAA (1 sample per pixel)
        self.multisampling.rasterization_samples = vk::SampleCountFlags::TYPE_1;
        self.multisampling.min_sample_shading = 1.0;
        self.multisampling.p_sample_mask = std::ptr::null();
        // no alpha to coverage either
        self.multisampling.alpha_to_coverage_enable = VK_FALSE;
        self.multisampling.alpha_to_one_enable = VK_FALSE;
    }

    pub(crate) fn disable_blending(&mut self) {
        // default write mask
        self.color_blend_attachment.color_write_mask = vk::ColorComponentFlags::RGBA;
        // no blending
        self.color_blend_attachment.blend_enable = VK_FALSE;
    }

    pub(crate) fn set_color_attachment_format(&mut self, format: vk::Format) {
        self.color_attachment_format = format;

        // connect to render info
        self.render_info.p_color_attachment_formats = &self.color_attachment_format;
        self.render_info.color_attachment_count = 1;
    }

    pub(crate) fn set_depth_format(&mut self, format: vk::Format) {
        self.render_info.depth_attachment_format = format;
    }

    pub(crate) fn disable_depth_test(&mut self) {
        self.depth_stencil.depth_test_enable = VK_FALSE;
        self.depth_stencil.depth_write_enable = VK_FALSE;
        self.depth_stencil.depth_compare_op = vk::CompareOp::NEVER;
        self.depth_stencil.depth_bounds_test_enable = VK_FALSE;
        self.depth_stencil.stencil_test_enable = VK_FALSE;
        self.depth_stencil.front = vk::StencilOpState::default();
        self.depth_stencil.back = vk::StencilOpState::default();
        self.depth_stencil.min_depth_bounds = 0.0;
        self.depth_stencil.max_depth_bounds = 1.0;
    }
}
