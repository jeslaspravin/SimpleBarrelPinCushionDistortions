# SimpleBarrelPinCushionDistortions
Test project that i did with Vulkan graphics API to Simulate Barrel Distortion and Pincushion distortion that will be used in VR for correcting lens distortions.

# Required Libraries<br>
1.LunarG® Vulkan™ SDK https://www.lunarg.com/vulkan-sdk/<br>
2.GLFW for Cross Platform windows creation http://www.glfw.org/<br>
3.GLM Mathematics Library https://glm.g-truc.net/0.9.9/index.html<br>
4.STB Image loaders https://github.com/nothings/stb<br>
5.Tiny Obj Loader https://github.com/syoyo/tinyobjloader<br><br>
Build system is generated for VS2015<br>

# Inputs<br>
W - To increase distortion upto max of 1.0(Max Barrel Distortion)<br>
S - To decrease distortion upto min of -1.0(Max Pincushion Distortion)<br>
T - To toggle between Normal mode(0 Distortion) and default mode(0.5 distortion)
