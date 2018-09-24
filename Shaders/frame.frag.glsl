#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0)out vec4 outColor;

layout(set=0,binding = 1) uniform sampler2DArray textureSampler;

layout(location = 0)in vec2 inFragCoord;

layout(set=0,binding =0) uniform UBO{
    layout(offset = 320) float distortionAlpha;
} ubo;

layout (constant_id = 0) const float LAYER_ID = 0.0f;

void main()
{
    const float alpha = ubo.distortionAlpha;

	vec2 p1 = vec2(2.0 * inFragCoord - 1.0);
	vec2 p2 = p1 / (1.0 - alpha * length(p1));
	p2 = (p2 + 1.0) * 0.5;

	bool inside = ((p2.x >= 0.0) && (p2.x <= 1.0) && (p2.y >= 0.0 ) && (p2.y <= 1.0));
	outColor = inside ? texture(textureSampler, vec3(p2, LAYER_ID)) : vec4(0.0);
}