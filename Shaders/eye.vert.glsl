#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_multiview : enable

out gl_PerVertex{
    vec4 gl_Position;
};

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragCoord;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 textureCoord;

layout(set=0,binding =0) uniform ProjectionData{
    mat4 modelTrans;
    mat4 viewTrans[2];
    mat4 projectionTrans[2];
} projectionTransforms;

void main()
{
    gl_Position=projectionTransforms.projectionTrans[gl_ViewIndex]* projectionTransforms.viewTrans[gl_ViewIndex]
     * projectionTransforms.modelTrans *vec4(inPosition,1.0);
    fragColor = inColor;
    fragCoord = textureCoord;
}