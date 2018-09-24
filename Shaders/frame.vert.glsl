#version 450
#extension GL_ARB_separate_shader_objects : enable

out gl_PerVertex{
    vec4 gl_Position;
};

layout(location = 0) out vec2 fragCoord;




void main()
{
    fragCoord = vec2((gl_VertexIndex<<1) & 2,gl_VertexIndex & 2);
    gl_Position = vec4(fragCoord*2 - 1.0,0.0,1.0);
}