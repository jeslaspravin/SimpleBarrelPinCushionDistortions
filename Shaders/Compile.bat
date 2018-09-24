@echo off
setlocal EnableDelayedExpansion

for %%f in (*.frag.glsl) do (

	E:/EduPrograms/Vulkan/1.1.82.1/Bin32/glslangValidator.exe -V %%f
	move /y "frag.spv" "%%~nf.spv"
)

for %%f in (*.vert.glsl) do (
	E:/EduPrograms/Vulkan/1.1.82.1/Bin32/glslangValidator.exe -V %%f
	move /y "vert.spv" "%%~nf.spv"
)


pause