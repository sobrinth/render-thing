DEL shader.vert.spv
DEL shader.frag.spv

%VULKAN_SDK%/Bin/glslangValidator.exe -V shader.vert
%VULKAN_SDK%/Bin/glslangValidator.exe -V shader.frag

REN vert.spv shader.vert.spv
REN frag.spv shader.frag.spv