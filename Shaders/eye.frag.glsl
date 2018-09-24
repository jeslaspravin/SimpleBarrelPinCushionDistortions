#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_multiview : enable

layout(location = 0)out vec4 outColor;

layout(set=1,binding = 0) uniform sampler2D textureSampler[2];

layout(location = 0)in vec3 inFragColor;
layout(location = 1)in vec2 inFragCoord;

void main()
{
    outColor=texture(textureSampler[gl_ViewIndex],inFragCoord);
}