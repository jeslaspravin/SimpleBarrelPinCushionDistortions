#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <memory>
#include <chrono>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/gtc/matrix_transform.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "types/VulkanTypes.h"
using namespace vulkan;

class RenderingApplication
{
private:
	// Handles
	GLFWwindow * window;

	VkInstance vulkanInstance = VK_NULL_HANDLE;

	VkSurfaceKHR surface = VK_NULL_HANDLE;

	VkPhysicalDevice vulkanDevice = VK_NULL_HANDLE;

	VkDevice logicalDevice = VK_NULL_HANDLE;

	VkSwapchainKHR swapChain = VK_NULL_HANDLE;

	VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

	VkQueue graphicsQueue = VK_NULL_HANDLE;
	VkQueue presentQueue = VK_NULL_HANDLE;
	VkQueue transferQueue = VK_NULL_HANDLE;

	// Informations
	VkExtent2D imageExtend;
	VkSurfaceFormatKHR choosenSurfaceFormat;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> swapChainframeBuffers;
	VkRenderPass renderPass;

	VkDescriptorSetLayout descriptorSetLayout;

	VkPipelineLayout pipelineLayout;
	VkPipeline pipeLine;

	// Vertex Buffer
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;

	// Indices 
	VkBuffer indicesBuffer;
	VkDeviceMemory indicesBufferMemory;

	// Uniform buffer
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	// Image buffers
	struct TextureData {
		uint32_t mipLevelsCount;
		VkImage textureImage;
		VkDeviceMemory textureImageMemory;
		VkImageView textureImageView;
		VkSampler textureSampler;
	};

	std::vector<TextureData> textures;

	VkSampleCountFlagBits msaaSampleBitsCount;

	VkImage depthTexture;
	VkDeviceMemory depthTextureMemory;
	VkImageView depthTextureImageView;
	VkFormat depthFormat;

	VkImage msaaColorRenderTarget;
	VkDeviceMemory colorRenderTargetMemory;
	VkImageView colorRenderTargetImageView;


	VkCommandPool graphicsCmdPool;
	std::vector<VkCommandBuffer> graphicsCmdBuffers;

	VkCommandPool transferCmdPool;

	// Synchronizing
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> imageRenderedSemaphores;
	std::vector<VkFence> fences;

	// Instance data
	int currentFrame;
	bool bIsWindowResized = false;

	std::vector<Vertex> vertices;

	std::vector<uint32_t> indices;

	// Multi view port data 
	
	VkRenderPass mvRenderPass;

	/*
	 *Depth Data
	 */
	VkImage mvDepthTexture;
	VkImageView mvDepthTextureImageView;
	VkDeviceMemory mvDepthTextureMemory;

	/*
	 *Color Texture
	 */
	VkImage mvColorTexture;
	VkImageView mvColorTextureImageView;
	VkDeviceMemory mvColorTextureMemory;
	VkSampler mvColorTextureSampler;

	VkFramebuffer mvFramebuffer;

	uint32_t noOfViews=2;

	VkPipelineLayout mvPipelineLayout;
	std::array<VkPipeline, 2> mvFramePipelines;

	VkDescriptorSetLayout textureDescriptorSetLayout;
	VkDescriptorSet textureDescriptorSet;

	VkFence mvTaskFence;
	VkSemaphore mvRenderingSemaphore;


	std::vector<VkCommandBuffer> mvCmdBuffers;

	float defaultDistortionAlpha = 0.5f;
	float currentDistAlpha;

	// Multi view port data ends

public:

	const int MAX_PARALLEL_FRAMES = 2;

	const uint16_t WND_WIDTH = 1280, WND_HEIGHT = 720;

	// Extensions that developer needs other than required Extensions for API Instance
	const std::vector<const char*> ADDITIONAL_INSTANCE_EXTENSIONS = {
#ifndef NDEBUG
		VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
		VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
	};

	const std::vector<const char*> ADDITIONAL_DEVICE_EXTENSIONS = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		"VK_KHX_multiview"
	};

	// List of layers to be enabled when instance creation
	const std::vector<const char*> LAYERS_TO_ENABLE = {
		"VK_LAYER_LUNARG_standard_validation"
	};

#ifdef NDEBUG
	bool bUseDebugMessenger = false;
#else
	bool bUseDebugMessenger = true;
#endif

public:
	void run()
	{
		initApp();
		mainLoop();
		cleanUp();
	}

private:

	void initApp()
	{
		currentDistAlpha = defaultDistortionAlpha;
		initGLFW();
		initVulkan();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(logicalDevice);
	}

	void cleanUp()
	{
		cleanSemaphores();

		vkDestroyCommandPool(logicalDevice, graphicsCmdPool, nullptr);
		vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
		vkFreeMemory(logicalDevice, vertexBufferMemory, nullptr);
		vkDestroyBuffer(logicalDevice, indicesBuffer, nullptr);
		vkFreeMemory(logicalDevice, indicesBufferMemory, nullptr);

		vkDestroySampler(logicalDevice, mvColorTextureSampler, nullptr);

		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
		cleanUniformBuffers();

		for (TextureData td : textures)
		{
			vkDestroySampler(logicalDevice, td.textureSampler, nullptr);
			vkDestroyImageView(logicalDevice, td.textureImageView, nullptr);
			vkDestroyImage(logicalDevice, td.textureImage, nullptr);
			vkFreeMemory(logicalDevice, td.textureImageMemory, nullptr);
		}

		vkDestroyCommandPool(logicalDevice, transferCmdPool, nullptr);

		cleanFrameBuffers(logicalDevice);
		cleanDepthResource();
		cleanImageResources();
		vkDestroyDescriptorSetLayout(logicalDevice, textureDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
		vkDestroyPipeline(logicalDevice, pipeLine, nullptr);
		for (uint32_t i = 0; i < noOfViews; i++)
		{
			vkDestroyPipeline(logicalDevice, mvFramePipelines[i], nullptr);
		}
		vkDestroyPipelineLayout(logicalDevice, mvPipelineLayout, nullptr);
		vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
		vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
		vkDestroyRenderPass(logicalDevice, mvRenderPass, nullptr);
		cleanImageViews();
		vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);

		vkDestroyDevice(logicalDevice, nullptr);

		if (bUseDebugMessenger)
			cleanDebugMessengerUtils();
		vkDestroySurfaceKHR(vulkanInstance, surface, nullptr);
		vkDestroyInstance(vulkanInstance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();

		vulkanInstance = nullptr;
		window = nullptr;
	}

	void initGLFW()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WND_WIDTH, WND_HEIGHT, "Vulkan Test", nullptr, nullptr);

		glfwSetKeyCallback(window, keyCallback);

		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, onWidowResize);
	}

	void initVulkan()
	{
		//loadModel(MDL_PATH);
		createCylinder(cylinderH, cylinderR, noOfSlices, cylinderAngle);
		createVulkanInstance();
		vulkan::VulkanTypes::setupNecessaryApi(vulkanInstance);
		if (bUseDebugMessenger)
			setupDebugMessengerUtils();

		createSurface();
		pickVulkanDevice();
		createLogicalDevice();
		createSwapChain();
		obtainImageAndImgViews();
		createRenderPass();
		createDescriptorLayout();
		createRenderPipeline();
		createCommandPool();

		createImageResources();
		createDepthResources();
		createFramebuffers();
		createVertexBuffers();
		createIndexBuffers();

		textures.resize(noOfViews);
		createImageTextureAndView("Textures/left.jpg",0);
		createImageTextureAndView("Textures/right.jpg", 1);
		createTextureSampler();

		createUniformBuffers();
		createDescriptorPool();
		allocDescriptorSets();

		allocAndRecordCmdBuffers();
		createSemaphores();
	}

	static void onWidowResize(GLFWwindow* window, int width, int height)
	{
		RenderingApplication* app = reinterpret_cast<RenderingApplication*>(glfwGetWindowUserPointer(window));
		app->bIsWindowResized = true;
	}

	void createVulkanInstance()
	{
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Test Render";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInstanceInfo = VkInstanceCreateInfo();
		createInstanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInstanceInfo.pApplicationInfo = &appInfo;

		uint32_t requiredExtensionCount;
		const char **requiredExtensionNames = glfwGetRequiredInstanceExtensions(&requiredExtensionCount);

		uint32_t sprtExtensionCount;
		vkEnumerateInstanceExtensionProperties(nullptr, &sprtExtensionCount, nullptr);
		std::vector<VkExtensionProperties> sprtExtensions(sprtExtensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &sprtExtensionCount, sprtExtensions.data());

		std::vector<const char*> useableExtensions = getAvailableExtensions(sprtExtensions, ADDITIONAL_INSTANCE_EXTENSIONS);

		std::vector<const char*> finalExtensionList(useableExtensions.size() + requiredExtensionCount);
		for (uint32_t i = 0; i < requiredExtensionCount; i++)
		{
			finalExtensionList[i] = requiredExtensionNames[i];
		}
		for (uint32_t i = 0; i < useableExtensions.size(); i++)
		{
			finalExtensionList[requiredExtensionCount + i] = useableExtensions[i];
		}

		createInstanceInfo.enabledExtensionCount = (uint32_t)finalExtensionList.size();
		createInstanceInfo.ppEnabledExtensionNames = finalExtensionList.data();

		std::vector<const char*> finalLayerList = {};
		if (bUseDebugMessenger)
		{
			uint32_t availableLayersCount;
			vkEnumerateInstanceLayerProperties(&availableLayersCount, nullptr);
			std::vector<VkLayerProperties> availableLayers(availableLayersCount);
			vkEnumerateInstanceLayerProperties(&availableLayersCount, availableLayers.data());

			finalLayerList = getAvailableLayers(availableLayers, LAYERS_TO_ENABLE);
		}

		createInstanceInfo.enabledLayerCount = (uint32_t)finalLayerList.size();
		createInstanceInfo.ppEnabledLayerNames = finalLayerList.data();

		VkResult result = vkCreateInstance(&createInstanceInfo, nullptr, &vulkanInstance);
		if (result == VK_SUCCESS)
		{
			std::cout << "Successfully created Vulkan Instance" << std::endl;
		}
		else
		{
			std::string errorMsg = "Failure in instantiating Vulkan!\n Cause : ";
			switch (result)
			{
			case VK_ERROR_LAYER_NOT_PRESENT:
				errorMsg.append("One/More of requested Vulkan layer/s is not available");
				break;
			case VK_ERROR_EXTENSION_NOT_PRESENT:
				errorMsg.append("One/More of requested Vulkan extension/s is not available");
				break;
			default:
				errorMsg.append("Other Reasons");
				break;
			}
			throw std::runtime_error(errorMsg.c_str());
		}

	}

	void createSurface()
	{
		if (glfwCreateWindowSurface(vulkanInstance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create Surface KHR from Window for process");
		}
	}

	std::vector<const char*> getAvailableExtensions(std::vector<VkExtensionProperties>& detectedExt, const std::vector<const char*>&
		queryingExtensions)
	{
		std::vector<const char*> returnData;
		const char* extensionChars = nullptr;
		std::string exts = "";
		for (const VkExtensionProperties& ext : detectedExt)
		{
			exts.append(ext.extensionName);
		}
		extensionChars = exts.c_str();

		for (const char* ext : queryingExtensions)
		{
			if (strstr(extensionChars, ext) != nullptr)
			{
				returnData.push_back(ext);
			}
			else
			{
				std::cerr << "Extension " << ext << " is not available" << std::endl;
			}
		}
		returnData.shrink_to_fit();
		return returnData;
	}

	std::vector<const char*> getAvailableLayers(std::vector<VkLayerProperties>& detectedLayers, const std::vector<const char*>&
		queryingLayers)
	{

		std::vector<const char*> returnData;
		const char* layerChars = nullptr;
		std::string layers = "";
		for (const VkLayerProperties& layer : detectedLayers)
		{
			layers.append(layer.layerName);
		}
		layerChars = layers.c_str();

		for (const char* layer : queryingLayers)
		{
			if (strstr(layerChars, layer) != nullptr)
			{
				returnData.push_back(layer);
			}
			else
			{
				std::cerr << "Layer " << layer << " is not available" << std::endl;
			}
		}
		returnData.shrink_to_fit();
		return returnData;
	}


	void pickVulkanDevice()
	{
		uint32_t supportedDeviceCount = 0;
		vkEnumeratePhysicalDevices(vulkanInstance, &supportedDeviceCount, nullptr);
		if (supportedDeviceCount == 0)
		{
			throw std::runtime_error("No Vulkan supported Graphics devices is available");
		}
		std::vector<VkPhysicalDevice> availableDevices(supportedDeviceCount);
		vkEnumeratePhysicalDevices(vulkanInstance, &supportedDeviceCount, availableDevices.data());
		for (const VkPhysicalDevice device : availableDevices)
		{
			if (isDeviceSuitable(device))
			{
				vulkanDevice = device;
				msaaSampleBitsCount = getMsaaSampleCount();
				break;
			}
		}

		if (vulkanDevice == VK_NULL_HANDLE)
		{
			std::runtime_error("Non of the available vulkan devices is suitable for the application");
		}
		else
		{
			VkPhysicalDeviceProperties deviceProps;
			VkPhysicalDeviceFeatures deviceFeatures;
			vkGetPhysicalDeviceProperties(vulkanDevice, &deviceProps);
			vkGetPhysicalDeviceFeatures(vulkanDevice, &deviceFeatures);
			std::cout << "Current GPU : " << deviceProps.deviceID << " - " << deviceProps.deviceName << std::endl;
			std::cout << "GPU Type : " << deviceProps.deviceType << std::endl;
			std::cout << "GPU Vendor ID : " << deviceProps.vendorID << std::endl;
			std::cout << "GPU Support - Float64 : " << (deviceFeatures.shaderFloat64 == 0 ? "False" : "True") << " Int16 : "
				<< (deviceFeatures.shaderInt16 == 0 ? "False" : "True") << std::endl;
		}
	}

	bool isDeviceSuitable(const VkPhysicalDevice& device)
	{
		VkPhysicalDeviceProperties deviceProps;
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceProperties(device, &deviceProps);
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		bool bHasSwapChainSupport = false;
		if (checkDeviceExtensionAvailability(device))
		{
			SwapChainSupport swapChainSupport = findSwapChainSupport(device);
			bHasSwapChainSupport = !(swapChainSupport.presentModes.empty() || swapChainSupport.surfaceFormats.empty());
		}

		return deviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && deviceFeatures.geometryShader && deviceFeatures.samplerAnisotropy
			&&	findQueueFamilyIndices(device).isComplete() && bHasSwapChainSupport;
	}

	QueueFamilyIndices findQueueFamilyIndices(VkPhysicalDevice device)
	{
		QueueFamilyIndices queueFamilyIndices = {};

		uint32_t availableQueueFamiliesCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &availableQueueFamiliesCount, nullptr);
		std::vector<VkQueueFamilyProperties> availableQueueFamilies(availableQueueFamiliesCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &availableQueueFamiliesCount, availableQueueFamilies.data());

		int i = 0;
		for (const VkQueueFamilyProperties& queueFamProps : availableQueueFamilies)
		{
			if (queueFamProps.queueCount > 0)
			{
				if (queueFamProps.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				{
					queueFamilyIndices.graphicsCmdQueue = i;
				}

				if (queueFamilyIndices.transferQueue == -1 && queueFamProps.queueFlags & VK_QUEUE_TRANSFER_BIT
					&& !(queueFamProps.queueFlags & VK_QUEUE_GRAPHICS_BIT))
				{
					queueFamilyIndices.transferQueue = i;
				}

				VkBool32 isPresentable = false;
				vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &isPresentable);
				if (isPresentable)
				{
					queueFamilyIndices.presentationCmdQueue = i;
				}
			}

			if (queueFamilyIndices.isComplete())
			{
				break;
			}
			i++;
		}

		if (queueFamilyIndices.transferQueue == -1)
		{
			queueFamilyIndices.transferQueue = queueFamilyIndices.graphicsCmdQueue;
		}
		return queueFamilyIndices;
	}

	bool checkDeviceExtensionAvailability(VkPhysicalDevice device)
	{
		uint32_t availableExtCount = 0;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &availableExtCount, nullptr);
		std::vector<VkExtensionProperties> availableExtensions(availableExtCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &availableExtCount, availableExtensions.data());

		std::vector<const char*> verifiedExts = getAvailableExtensions(availableExtensions, ADDITIONAL_DEVICE_EXTENSIONS);
		return verifiedExts.size() == ADDITIONAL_DEVICE_EXTENSIONS.size();
	}

	SwapChainSupport findSwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupport support;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &support.surfaceCapabilities);

		uint32_t count = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr);
		if (count > 0)
		{
			support.surfaceFormats.resize(count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, support.surfaceFormats.data());
		}

		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
		if (count > 0)
		{
			support.presentModes.resize(count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, support.presentModes.data());
		}

		return support;
	}

	VkExtent2D chooseSwapExtent(VkSurfaceCapabilitiesKHR &surfaceCapabilities)
	{
		if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{

			std::cout << "Current Frame buffer " << surfaceCapabilities.currentExtent.width << "x" <<
				surfaceCapabilities.currentExtent.height << std::endl;
			return surfaceCapabilities.currentExtent;
		}
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D extent = { static_cast<uint32_t>(width) ,static_cast<uint32_t>(height) };

			std::cout << "Current Frame buffer " << width << "x" << height << std::endl;

			extent.width = std::max(surfaceCapabilities.minImageExtent.width,
				std::min(surfaceCapabilities.maxImageExtent.width, extent.width));
			extent.height = std::max(surfaceCapabilities.minImageExtent.height,
				std::min(surfaceCapabilities.maxImageExtent.height, extent.height));
			return extent;
		}
	}

	VkFormat chooseImageFormat(const std::vector<VkFormat>& checkFormats, VkImageTiling imageTiling, VkFormatFeatureFlags formatFeatureFlag)
	{
		for (const VkFormat format : checkFormats)
		{
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(vulkanDevice, format, &props);

			if (imageTiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & formatFeatureFlag) == formatFeatureFlag)
			{
				return format;
			}
			else if (imageTiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & formatFeatureFlag) == formatFeatureFlag)
			{
				return format;
			}
		}

		throw std::runtime_error("None of requested formats supports necessary features for given tiling in this hardware");
	}

	VkFormat chooseDepthImageFormat()
	{
		return chooseImageFormat({ VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D32_SFLOAT,VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	bool hasStencilFormat(VkFormat format)
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void createLogicalDevice() {
		QueueFamilyIndices queueIndices = findQueueFamilyIndices(vulkanDevice);

		std::vector<VkDeviceQueueCreateInfo> allQueueCreateInfo;
		std::set<int> uniqueQueueIndex = { queueIndices.graphicsCmdQueue,queueIndices.presentationCmdQueue,queueIndices.transferQueue };

		float queuePriority = 1.f;

		for (int queueIdx : uniqueQueueIndex)
		{
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.queueFamilyIndex = queueIdx;

			queueCreateInfo.pQueuePriorities = &queuePriority;

			allQueueCreateInfo.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		// Just do get supported extensions of device and check with required extensions when using one
		deviceCreateInfo.enabledExtensionCount = (uint32_t)ADDITIONAL_DEVICE_EXTENSIONS.size();
		deviceCreateInfo.ppEnabledExtensionNames = ADDITIONAL_DEVICE_EXTENSIONS.data();

		//Start Layers to Enable for logical device
		std::vector<const char*> finalLayerList = {};
		if (bUseDebugMessenger)
		{
			uint32_t availableLayersCount;
			vkEnumerateDeviceLayerProperties(vulkanDevice, &availableLayersCount, nullptr);
			std::vector<VkLayerProperties> availableLayers(availableLayersCount);
			vkEnumerateDeviceLayerProperties(vulkanDevice, &availableLayersCount, availableLayers.data());

			finalLayerList = getAvailableLayers(availableLayers, LAYERS_TO_ENABLE);
		}

		deviceCreateInfo.enabledLayerCount = (uint32_t)finalLayerList.size();
		deviceCreateInfo.ppEnabledLayerNames = finalLayerList.data();
		// End Layers
		deviceCreateInfo.pQueueCreateInfos = allQueueCreateInfo.data();
		deviceCreateInfo.queueCreateInfoCount = (uint32_t)allQueueCreateInfo.size();

		deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
		if (vkCreateDevice(vulkanDevice, &deviceCreateInfo, nullptr, &logicalDevice) != VK_SUCCESS)
		{
			throw std::runtime_error("Unable to create logical device");
		}

		vkGetDeviceQueue(logicalDevice, queueIndices.graphicsCmdQueue, 0, &graphicsQueue);
		vkGetDeviceQueue(logicalDevice, queueIndices.presentationCmdQueue, 0, &presentQueue);
		vkGetDeviceQueue(logicalDevice, queueIndices.transferQueue, 0, &transferQueue);
	}

	void setupDebugMessengerUtils()
	{
		if (vulkan::VulkanTypes::fnVkCreateDebugUtilsMessengerExt != nullptr)
		{
			VkDebugUtilsMessengerCreateInfoEXT messengerInfo = {};
			messengerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			messengerInfo.pfnUserCallback = debugCallback;
			messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
			messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
				| VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			vulkan::VulkanTypes::fnVkCreateDebugUtilsMessengerExt(vulkanInstance, &messengerInfo, nullptr, &debugMessenger);
		}
		else
		{
			throw std::runtime_error("Debug Messenger utilities function is not available");
		}
	}

	void createSwapChain()
	{
		SwapChainSupport swapChainSupport = findSwapChainSupport(vulkanDevice);
		choosenSurfaceFormat = swapChainSupport.chooseSurfaceFormat();
		VkPresentModeKHR presentMode = swapChainSupport.choosePresentMode();
		imageExtend = chooseSwapExtent(swapChainSupport.surfaceCapabilities);
		uint32_t imageCount = swapChainSupport.surfaceCapabilities.minImageCount + 1;
		imageCount = swapChainSupport.surfaceCapabilities.maxImageCount > 0 ?
			std::min(swapChainSupport.surfaceCapabilities.maxImageCount, imageCount) : imageCount;



		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.clipped = VK_TRUE;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.imageArrayLayers = 1;
		createInfo.imageColorSpace = choosenSurfaceFormat.colorSpace;
		createInfo.imageFormat = choosenSurfaceFormat.format;
		createInfo.imageExtent = imageExtend;
		createInfo.minImageCount = imageCount;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		createInfo.presentMode = presentMode;
		createInfo.preTransform = swapChainSupport.surfaceCapabilities.currentTransform;// Use necessary flags if needed advanced operations
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		QueueFamilyIndices queuesRequired = findQueueFamilyIndices(vulkanDevice);
		uint32_t queues[] = { (uint32_t)queuesRequired.graphicsCmdQueue,(uint32_t)queuesRequired.presentationCmdQueue };
		if (queuesRequired.graphicsCmdQueue != queuesRequired.presentationCmdQueue)
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queues;
		}
		else
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create swap chain to draw images to!");
		}
	}

	// Get the created images to draw and store handle to it
	void obtainImageAndImgViews()
	{
		uint32_t imagesCount;
		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imagesCount, nullptr);
		swapChainImages.resize(imagesCount);
		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imagesCount, swapChainImages.data());

		swapChainImageViews.resize(imagesCount);

		for (size_t i = 0; i < imagesCount; i++)
		{
			createImageView(swapChainImages[i], 1, choosenSurfaceFormat.format, VK_IMAGE_ASPECT_COLOR_BIT, swapChainImageViews[i]);
		}

	}

	void cleanImageViews()
	{
		for (VkImageView &imgView : swapChainImageViews)
		{
			vkDestroyImageView(logicalDevice, imgView, nullptr);
		}
		swapChainImageViews.clear();
	}

	void cleanDebugMessengerUtils()
	{
		if (vulkan::VulkanTypes::fnVkDestroyDebugUtilsMessengerExt != nullptr)
		{
			vulkan::VulkanTypes::fnVkDestroyDebugUtilsMessengerExt(vulkanInstance, debugMessenger, nullptr);
		}
	}

	void createRenderPass()
	{
		VkAttachmentDescription colorAttachmentDesc = {};
		colorAttachmentDesc.format = choosenSurfaceFormat.format;
		colorAttachmentDesc.samples = msaaSampleBitsCount;

		colorAttachmentDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachmentDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

		colorAttachmentDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		colorAttachmentDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription depthAttachmentDesc = {};
		depthAttachmentDesc.format = (depthFormat = chooseDepthImageFormat());
		depthAttachmentDesc.samples = msaaSampleBitsCount;

		depthAttachmentDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachmentDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

		depthAttachmentDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachmentDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		depthAttachmentDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachmentDesc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription colorAttachmentResolveDesc = {};
		colorAttachmentResolveDesc.format = choosenSurfaceFormat.format;
		colorAttachmentResolveDesc.samples = VK_SAMPLE_COUNT_1_BIT;

		colorAttachmentResolveDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachmentResolveDesc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		colorAttachmentResolveDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolveDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		colorAttachmentResolveDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolveDesc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		// Attachment references for subpasses
		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef = {};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentResolveRef = {};
		colorAttachmentResolveRef.attachment = 2;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// Subpass 
		VkSubpassDescription subpassDesc = {};
		subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDesc.colorAttachmentCount = 1;
		subpassDesc.pColorAttachments = &colorAttachmentRef;
		subpassDesc.pDepthStencilAttachment = &depthAttachmentRef;
		subpassDesc.pResolveAttachments = &colorAttachmentResolveRef;

		// Render pass
		VkRenderPassCreateInfo renderPassCreateInfo = {};
		renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpassDesc;

		std::array<VkAttachmentDescription, 3> attachments = { colorAttachmentDesc,depthAttachmentDesc,colorAttachmentResolveDesc };

		renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassCreateInfo.pAttachments = attachments.data();

		VkSubpassDependency dependencies = {};
		dependencies.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies.dstSubpass = 0;
		dependencies.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies.srcAccessMask = 0;
		dependencies.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		renderPassCreateInfo.dependencyCount = 1;
		renderPassCreateInfo.pDependencies = &dependencies;

		if (vkCreateRenderPass(logicalDevice, &renderPassCreateInfo, nullptr, &renderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed creating render pass for current frame rendering");
		}

		// Render pass for multi view

		colorAttachmentDesc.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		colorAttachmentDesc.samples = VK_SAMPLE_COUNT_1_BIT;

		depthAttachmentDesc.samples = VK_SAMPLE_COUNT_1_BIT;

		subpassDesc.pResolveAttachments = nullptr;

		std::array<VkAttachmentDescription, 2> multiViewAttachments = { colorAttachmentDesc,depthAttachmentDesc };

		renderPassCreateInfo.attachmentCount = (uint32_t)multiViewAttachments.size();
		renderPassCreateInfo.pAttachments = multiViewAttachments.data();

		const uint32_t viewMask = 0b00000011;
		const uint32_t correlationMask = 0b00000011;

		VkRenderPassMultiviewCreateInfo multiViewRenderPassCI = {};
		multiViewRenderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
		multiViewRenderPassCI.subpassCount = 1;
		multiViewRenderPassCI.correlationMaskCount = 1;
		multiViewRenderPassCI.pCorrelationMasks = &correlationMask;
		multiViewRenderPassCI.pViewMasks = &viewMask;

		renderPassCreateInfo.pNext = &multiViewRenderPassCI;

		if (vkCreateRenderPass(logicalDevice, &renderPassCreateInfo, nullptr, &mvRenderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed creating render pass for multiview port rendering");
		}

	}

	void createDescriptorLayout()
	{
		VkDescriptorSetLayoutBinding descriptorUboLayoutBind = {};
		descriptorUboLayoutBind.binding = 0;
		descriptorUboLayoutBind.descriptorCount = 1;
		descriptorUboLayoutBind.pImmutableSamplers = nullptr;
		descriptorUboLayoutBind.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorUboLayoutBind.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding descriptorImageLayoutBind = {};
		descriptorImageLayoutBind.binding = 1;
		descriptorImageLayoutBind.descriptorCount = 1;
		descriptorImageLayoutBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorImageLayoutBind.pImmutableSamplers = nullptr;
		descriptorImageLayoutBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> descriptors = { descriptorUboLayoutBind ,descriptorImageLayoutBind };

		VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo = {};
		descriptorLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorLayoutCreateInfo.bindingCount = static_cast<uint32_t>(descriptors.size());
		descriptorLayoutCreateInfo.pBindings = descriptors.data();

		if (vkCreateDescriptorSetLayout(logicalDevice, &descriptorLayoutCreateInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failure in creating Descriptor Set Layout");
		}


		descriptorImageLayoutBind.binding = 0;
		descriptorImageLayoutBind.descriptorCount = noOfViews;

		descriptorLayoutCreateInfo.bindingCount = 1;
		descriptorLayoutCreateInfo.pBindings = &descriptorImageLayoutBind;

		if (vkCreateDescriptorSetLayout(logicalDevice, &descriptorLayoutCreateInfo, nullptr, &textureDescriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failure in creating Descriptor Set Layout for texture alone");
		}
	}

	void createRenderPipeline()
	{
		// Multiview pipeline
		// Start : Programmable section of pipeline

		std::vector<char> fragShaderCode = readShaderFile("Shaders/eye.frag.spv");
		std::vector<char> vertShaderCode = readShaderFile("Shaders/eye.vert.spv");

		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);

		VkPipelineShaderStageCreateInfo fragShaderCreateInfo = {};
		fragShaderCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderCreateInfo.pName = "main";
		fragShaderCreateInfo.module = fragShaderModule;

		VkPipelineShaderStageCreateInfo vertShaderCreateInfo = {};
		vertShaderCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderCreateInfo.pName = "main";
		vertShaderCreateInfo.module = vertShaderModule;

		VkPipelineShaderStageCreateInfo pipelineShaderStages[] = { fragShaderCreateInfo,vertShaderCreateInfo };

		// End : Programmable section of pipeline

		// Start : Fixed functions

		// 1 Vertex Input defining stage
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		auto vertexAttribDesc = Vertex::getAttributeDesc();
		auto vertexBindDesc = Vertex::getBindingDesc();
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttribDesc.size());
		vertexInputInfo.pVertexAttributeDescriptions = vertexAttribDesc.data();
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &vertexBindDesc;

		// 2 Input Assembly stage
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = {};
		inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

		// 3 Viewport and Scissor Rectangle 
		VkViewport viewport = {};
		viewport.x = viewport.y = 0;
		viewport.width = (float)imageExtend.width;
		viewport.height = (float)imageExtend.height;
		viewport.maxDepth = 1;
		viewport.minDepth = 0;

		VkRect2D scissorRect = {};
		scissorRect.extent = imageExtend;
		scissorRect.offset = { 0,0 };

		VkPipelineViewportStateCreateInfo viewportCreateInfo = {};
		viewportCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportCreateInfo.scissorCount = 1;
		viewportCreateInfo.pScissors = &scissorRect;
		viewportCreateInfo.viewportCount = 1;
		viewportCreateInfo.pViewports = &viewport;

		// 4 Rasterization stage
		VkPipelineRasterizationStateCreateInfo rasterizationCreateInfo = {};
		rasterizationCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizationCreateInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterizationCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationCreateInfo.depthClampEnable = VK_FALSE;
		rasterizationCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizationCreateInfo.lineWidth = 1.0f;
		rasterizationCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizationCreateInfo.depthBiasEnable = VK_FALSE;
		rasterizationCreateInfo.depthBiasConstantFactor = rasterizationCreateInfo.depthBiasSlopeFactor = rasterizationCreateInfo.depthBiasClamp = 0.f;

		// 5 Multisampling 
		VkPipelineMultisampleStateCreateInfo multisamplingCreateInfo = {};
		multisamplingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisamplingCreateInfo.alphaToCoverageEnable = VK_FALSE;
		multisamplingCreateInfo.alphaToOneEnable = VK_FALSE;
		multisamplingCreateInfo.minSampleShading = 1.0f;
		multisamplingCreateInfo.pSampleMask = nullptr;
		multisamplingCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// 6 Stencils and Depth Test
		VkPipelineDepthStencilStateCreateInfo depthStensilCreateInfo = {};
		depthStensilCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

		depthStensilCreateInfo.stencilTestEnable = VK_FALSE;
		depthStensilCreateInfo.front = {};
		depthStensilCreateInfo.back = {};

		depthStensilCreateInfo.depthBoundsTestEnable = VK_FALSE;
		depthStensilCreateInfo.maxDepthBounds = 1.0f;
		depthStensilCreateInfo.minDepthBounds = 0.0f;

		depthStensilCreateInfo.depthTestEnable = VK_TRUE;
		depthStensilCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStensilCreateInfo.depthWriteEnable = VK_TRUE;

		// 7 Color Blending
		VkPipelineColorBlendAttachmentState attachmentState = {};
		attachmentState.blendEnable = VK_FALSE;
		attachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		attachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		attachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		attachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		attachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		attachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		attachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

		VkPipelineColorBlendStateCreateInfo blendStateInfo = {};
		blendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendStateInfo.attachmentCount = 1;
		blendStateInfo.pAttachments = &attachmentState;
		blendStateInfo.logicOpEnable = VK_FALSE;
		blendStateInfo.logicOp = VK_LOGIC_OP_COPY;
		blendStateInfo.blendConstants[0] = blendStateInfo.blendConstants[1] = blendStateInfo.blendConstants[2] = blendStateInfo.blendConstants[3] = 0.f;

		// 8 Dynamic states 
		// TODO : Later
		// 

		// 9 Pipeline Layout creation

		VkDescriptorSetLayout layouts[2] = { descriptorSetLayout , textureDescriptorSetLayout };

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
		pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
		pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;
		pipelineLayoutCreateInfo.setLayoutCount = 2;
		pipelineLayoutCreateInfo.pSetLayouts = layouts;

		if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create pipeline layout");
		}

		// End : Fixed functions

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
		pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineCreateInfo.pTessellationState = nullptr;
		pipelineCreateInfo.pViewportState = &viewportCreateInfo;
		pipelineCreateInfo.pRasterizationState = &rasterizationCreateInfo;
		pipelineCreateInfo.pMultisampleState = &multisamplingCreateInfo;
		pipelineCreateInfo.pDepthStencilState = &depthStensilCreateInfo;
		pipelineCreateInfo.pDynamicState = nullptr;
		pipelineCreateInfo.pColorBlendState = &blendStateInfo;

		pipelineCreateInfo.layout = pipelineLayout;
		pipelineCreateInfo.stageCount = 2;
		pipelineCreateInfo.pStages = pipelineShaderStages;
		pipelineCreateInfo.renderPass = mvRenderPass;
		pipelineCreateInfo.subpass = 0;

		pipelineCreateInfo.basePipelineHandle = nullptr;
		pipelineCreateInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, nullptr, 1, &pipelineCreateInfo, nullptr, &pipeLine) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed creating graphics pipeline");
		}

		vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
		vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);



		// Frame Rendering Pipeline

		float multiviewArrayLayer = 0.0f;

		VkSpecializationMapEntry specializationMapEntry = {0,0,sizeof(float)};

		VkSpecializationInfo specializationInfo = {};
		specializationInfo.dataSize = sizeof(float);
		specializationInfo.mapEntryCount = 1;
		specializationInfo.pMapEntries = &specializationMapEntry;
		specializationInfo.pData = &multiviewArrayLayer;

		rasterizationCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;

		// Only UBO and Rendered Targets Descriptors are needed for final frame pipeline passes
		pipelineLayoutCreateInfo.setLayoutCount = 1;
		pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

		if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutCreateInfo, nullptr, &mvPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create pipeline layout");
		}

		for (uint32_t i = 0; i < noOfViews; i++)
		{
			fragShaderCode = readShaderFile("Shaders/frame.frag.spv");
			vertShaderCode = readShaderFile("Shaders/frame.vert.spv");
			fragShaderModule = createShaderModule(fragShaderCode);
			vertShaderModule = createShaderModule(vertShaderCode);
			fragShaderCreateInfo.module = fragShaderModule;
			vertShaderCreateInfo.module = vertShaderModule;
			fragShaderCreateInfo.pSpecializationInfo = &specializationInfo;

			VkPipelineShaderStageCreateInfo frameShaderStages[] = { fragShaderCreateInfo,vertShaderCreateInfo };
			
			VkPipelineVertexInputStateCreateInfo inputStateInfo = {};
			inputStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			
			std::array<VkDynamicState, 2> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR };
			VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
			dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
			dynamicStateInfo.dynamicStateCount = (uint32_t)dynamicStates.size();
			dynamicStateInfo.pDynamicStates = dynamicStates.data();

			multisamplingCreateInfo.rasterizationSamples = msaaSampleBitsCount;


			pipelineCreateInfo.stageCount = 2;
			pipelineCreateInfo.pRasterizationState = &rasterizationCreateInfo;
			pipelineCreateInfo.pStages = frameShaderStages;
			pipelineCreateInfo.pVertexInputState = &inputStateInfo;
			pipelineCreateInfo.pDynamicState = &dynamicStateInfo;
			pipelineCreateInfo.pMultisampleState = &multisamplingCreateInfo;
			pipelineCreateInfo.layout = mvPipelineLayout;
			pipelineCreateInfo.renderPass = renderPass;

			if (vkCreateGraphicsPipelines(logicalDevice, nullptr, 1, &pipelineCreateInfo, nullptr, &(mvFramePipelines[i])) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed creating frame graphics pipeline");
			}

			multiviewArrayLayer++;


			vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
			vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
		}

	}

	// Creates framebuffer to be used with render pass in command buffers
	void createFramebuffers()
	{
		swapChainframeBuffers.resize(swapChainImageViews.size());

		for (int i = 0; i < swapChainframeBuffers.size(); i++)
		{
			std::array<VkImageView, 3> imgViews = {
				colorRenderTargetImageView,
				depthTextureImageView,
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo frameBufferCreateInfo = {};
			frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frameBufferCreateInfo.renderPass = renderPass;
			frameBufferCreateInfo.layers = 1;
			frameBufferCreateInfo.width = imageExtend.width;
			frameBufferCreateInfo.height = imageExtend.height;
			frameBufferCreateInfo.attachmentCount = static_cast<uint32_t>(imgViews.size());
			frameBufferCreateInfo.pAttachments = imgViews.data();

			if (vkCreateFramebuffer(logicalDevice, &frameBufferCreateInfo, nullptr, &swapChainframeBuffers[i]) != VK_SUCCESS)
			{
				std::string fmt = "Failed creating frame buffer for image view at index %d";
				size_t size = snprintf(nullptr, 0, fmt.c_str(), i) + 1;
				std::unique_ptr<char[]> buffer(new char[size]);
				snprintf(buffer.get(), size, fmt.c_str(), i);
				throw std::runtime_error(buffer.get());
			}
		}

		std::array<VkImageView, 2> imgViews = {
			mvColorTextureImageView,
			mvDepthTextureImageView
		};

		VkFramebufferCreateInfo fbCreateInfo = {};
		fbCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbCreateInfo.attachmentCount =(uint32_t)imgViews.size();
		fbCreateInfo.height = imageExtend.height;
		fbCreateInfo.width = imageExtend.width;
		fbCreateInfo.pAttachments = imgViews.data();
		fbCreateInfo.layers = 1;
		fbCreateInfo.renderPass = mvRenderPass;

		if (vkCreateFramebuffer(logicalDevice, &fbCreateInfo, nullptr, &mvFramebuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed creating framebuffer for multiview");
		}
	}

	void cleanFrameBuffers(VkDevice device)
	{
		for (VkFramebuffer &framebuffer : swapChainframeBuffers)
		{
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		swapChainframeBuffers.clear();
		vkDestroyFramebuffer(device, mvFramebuffer, nullptr);
	}

	void createVertexBuffers()
	{
		size_t size = sizeof(vertices[0])*vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBufferMemory(size, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory);

		void *data;

		vkMapMemory(logicalDevice, stagingBufferMemory, 0, size, 0, &data);
		memcpy(data, vertices.data(), size);
		vkUnmapMemory(logicalDevice, stagingBufferMemory);

		createBufferMemory(size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
			, vertexBuffer, vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, size);

		vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

	}


	void createIndexBuffers()
	{
		VkDeviceSize size = sizeof(indices[0])*indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBufferMemory(size, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			stagingBuffer, stagingBufferMemory);

		void *data;

		vkMapMemory(logicalDevice, stagingBufferMemory, 0, size, 0, &data);
		memcpy(data, indices.data(), size);
		vkUnmapMemory(logicalDevice, stagingBufferMemory);

		createBufferMemory(size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
			, indicesBuffer, indicesBufferMemory);

		copyBuffer(stagingBuffer, indicesBuffer, size);

		vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
	}


	void createUniformBuffers()
	{
		VkDeviceSize size = sizeof(ProjectionData);

		uniformBuffers.resize(swapChainImageViews.size());
		uniformBuffersMemory.resize(swapChainImageViews.size());

		for (int i = 0; i < swapChainImageViews.size(); i++)
		{
			createBufferMemory(size, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
		}
	}

	void cleanUniformBuffers()
	{
		for (int i = 0; i < swapChainImageViews.size(); i++)
		{
			vkDestroyBuffer(logicalDevice, uniformBuffers[i], nullptr);
			vkFreeMemory(logicalDevice, uniformBuffersMemory[i], nullptr);
		}
	}


	void createDescriptorPool()
	{

		std::array<VkDescriptorPoolSize, 3> poolSizes;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImageViews.size());
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImageViews.size());
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[2].descriptorCount = 2;
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

		VkDescriptorPoolCreateInfo descPoolCreateInfo = {};
		descPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		descPoolCreateInfo.pPoolSizes = poolSizes.data();
		descPoolCreateInfo.maxSets = static_cast<uint32_t>(swapChainImageViews.size()+1);

		if (vkCreateDescriptorPool(logicalDevice, &descPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("Failure in creating Descriptor Set Pool");
		}
	}

	void allocDescriptorSets()
	{
		{
			std::vector<VkDescriptorSetLayout> layouts(swapChainImageViews.size(), descriptorSetLayout);

			VkDescriptorSetAllocateInfo allocateInfo = {};
			allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocateInfo.descriptorPool = descriptorPool;
			allocateInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImageViews.size());
			allocateInfo.pSetLayouts = layouts.data();

			descriptorSets.resize(swapChainImageViews.size());

			if (vkAllocateDescriptorSets(logicalDevice, &allocateInfo, descriptorSets.data()) != VK_SUCCESS)
			{
				throw std::runtime_error("Unable to allocate Descriptor Sets for UBO and Sample Texture from Pool");
			}

			for (int i = 0; i < swapChainImageViews.size(); i++)
			{
				VkDescriptorBufferInfo descBufferInfo = {};
				descBufferInfo.buffer = uniformBuffers[i];
				descBufferInfo.offset = 0;
				descBufferInfo.range = sizeof(ProjectionData);

				VkDescriptorImageInfo descImageInfo = {};
				descImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				descImageInfo.imageView = mvColorTextureImageView;
				descImageInfo.sampler = mvColorTextureSampler;

				VkWriteDescriptorSet bufferWriteDescriptorSet = {};
				bufferWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				bufferWriteDescriptorSet.descriptorCount = 1;
				bufferWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				bufferWriteDescriptorSet.dstBinding = 0;
				bufferWriteDescriptorSet.dstArrayElement = 0;
				bufferWriteDescriptorSet.dstSet = descriptorSets[i];
				bufferWriteDescriptorSet.pBufferInfo = &descBufferInfo;
				bufferWriteDescriptorSet.pImageInfo = nullptr;
				bufferWriteDescriptorSet.pTexelBufferView = nullptr;

				VkWriteDescriptorSet imageWriteDescriptorSet = {};
				imageWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				imageWriteDescriptorSet.descriptorCount = 1;
				imageWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				imageWriteDescriptorSet.dstBinding = 1;
				imageWriteDescriptorSet.dstArrayElement = 0;
				imageWriteDescriptorSet.dstSet = descriptorSets[i];
				imageWriteDescriptorSet.pBufferInfo = nullptr;
				imageWriteDescriptorSet.pImageInfo = &descImageInfo;
				imageWriteDescriptorSet.pTexelBufferView = nullptr;

				std::array<VkWriteDescriptorSet, 2> writeDescriptorSets = { bufferWriteDescriptorSet ,imageWriteDescriptorSet };

				vkUpdateDescriptorSets(logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(),
					0, nullptr);
			}
		}

		{
			// Made all to descriptor sets to 1 as they are not needed


			VkDescriptorSetAllocateInfo allocateInfo = {};
			allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocateInfo.descriptorPool = descriptorPool;
			allocateInfo.descriptorSetCount = 1;
			allocateInfo.pSetLayouts = &textureDescriptorSetLayout;



			if (vkAllocateDescriptorSets(logicalDevice, &allocateInfo, &textureDescriptorSet) != VK_SUCCESS)
			{
				throw std::runtime_error("Unable to allocate Descriptor Sets for textures from Pool");
			}

			//for (int i = 0; i < 1; i++)
			//{
				VkDescriptorImageInfo descImageInfo = {};
				descImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				descImageInfo.imageView = textures[0].textureImageView;
				descImageInfo.sampler = textures[0].textureSampler;

				VkDescriptorImageInfo descImageInfo01 = {};
				descImageInfo01.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				descImageInfo01.imageView = textures[1].textureImageView;
				descImageInfo01.sampler = textures[1].textureSampler;

				VkWriteDescriptorSet imageWriteDescriptorSet = {};
				imageWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				imageWriteDescriptorSet.descriptorCount = 1;
				imageWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				imageWriteDescriptorSet.dstBinding = 0;
				imageWriteDescriptorSet.dstArrayElement = 0;
				imageWriteDescriptorSet.dstSet = textureDescriptorSet;
				imageWriteDescriptorSet.pBufferInfo = nullptr;
				imageWriteDescriptorSet.pImageInfo = &descImageInfo;
				imageWriteDescriptorSet.pTexelBufferView = nullptr;

				VkWriteDescriptorSet imageWriteDescriptorSet01 = {};
				imageWriteDescriptorSet01.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				imageWriteDescriptorSet01.descriptorCount = 1;
				imageWriteDescriptorSet01.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				imageWriteDescriptorSet01.dstBinding = 0;
				imageWriteDescriptorSet01.dstArrayElement = 1;
				imageWriteDescriptorSet01.dstSet = textureDescriptorSet;
				imageWriteDescriptorSet01.pBufferInfo = nullptr;
				imageWriteDescriptorSet01.pImageInfo = &descImageInfo01;
				imageWriteDescriptorSet01.pTexelBufferView = nullptr;

				std::array<VkWriteDescriptorSet, 2> writeImages = { imageWriteDescriptorSet ,imageWriteDescriptorSet01 };

				vkUpdateDescriptorSets(logicalDevice, (uint32_t)writeImages.size(), writeImages.data(),0, nullptr);
			//}
		}


	}

	uint32_t chooseMemoryType(uint32_t filterMemType, VkMemoryPropertyFlags propertyFlags)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(vulkanDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if (filterMemType & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & propertyFlags) == propertyFlags)
			{
				return i;
			}
		}

		throw std::runtime_error("No suitable memory type is available for Vertex buffer");
	}

	void createCommandPool()
	{
		QueueFamilyIndices queueFamilies = findQueueFamilyIndices(vulkanDevice);

		VkCommandPoolCreateInfo cmdPoolCreateInfo = {};
		cmdPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolCreateInfo.queueFamilyIndex = queueFamilies.graphicsCmdQueue;
		cmdPoolCreateInfo.flags = 0;

		if (vkCreateCommandPool(logicalDevice, &cmdPoolCreateInfo, nullptr, &graphicsCmdPool) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed creating Graphics Command Pool");
		}

		if (queueFamilies.transferQueue != queueFamilies.graphicsCmdQueue)
		{
			cmdPoolCreateInfo.queueFamilyIndex = queueFamilies.transferQueue;

			if (vkCreateCommandPool(logicalDevice, &cmdPoolCreateInfo, nullptr, &transferCmdPool) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed creating Buffer Transfer Command Pool");
			}
		}
		else
		{
			transferCmdPool = graphicsCmdPool;
		}
	}

	void allocAndRecordCmdBuffers()
	{
		graphicsCmdBuffers.resize(swapChainframeBuffers.size());
		VkCommandBufferAllocateInfo cmdBufferAllocInfo = {};
		cmdBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufferAllocInfo.commandBufferCount = (uint32_t)graphicsCmdBuffers.size();
		cmdBufferAllocInfo.commandPool = graphicsCmdPool;
		cmdBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(logicalDevice, &cmdBufferAllocInfo, graphicsCmdBuffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allocate command buffers");
		}

		for (int i = 0; i < graphicsCmdBuffers.size(); i++)
		{
			VkCommandBufferBeginInfo cmdBuffBeginInfo = {};
			cmdBuffBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			cmdBuffBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
			cmdBuffBeginInfo.pInheritanceInfo = nullptr;

			if (vkBeginCommandBuffer(graphicsCmdBuffers[i], &cmdBuffBeginInfo) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to begin command buffer");
			}

			VkRenderPassBeginInfo renderPassBeginInfo = {};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.renderPass = mvRenderPass;
			renderPassBeginInfo.framebuffer = mvFramebuffer;
			renderPassBeginInfo.renderArea.offset = { 0,0 };
			renderPassBeginInfo.renderArea.extent = imageExtend;

			std::array<VkClearValue, 2> clearVals = {};
			clearVals[0].color = { 0.0f, 0.0f, 0.0f, 1.f };
			clearVals[1].depthStencil = { 1.0f,0 };
			//clearVals[2].color = { 0.0f, 0.0f, 0.0f, 1.f };

			renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearVals.size());
			renderPassBeginInfo.pClearValues = clearVals.data();

			vkCmdBeginRenderPass(graphicsCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(graphicsCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLine);

			VkBuffer vertexBuffers[] = { vertexBuffer };
			VkDeviceSize bufferOffsets[] = { 0 };

			vkCmdBindVertexBuffers(graphicsCmdBuffers[i], 0, 1, vertexBuffers, bufferOffsets);
			vkCmdBindIndexBuffer(graphicsCmdBuffers[i], indicesBuffer, 0, VK_INDEX_TYPE_UINT32);

			std::array<VkDescriptorSet, 2> descSets = { descriptorSets[i] ,textureDescriptorSet};
			vkCmdBindDescriptorSets(graphicsCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, noOfViews,
				descSets.data(), 0, nullptr);

			//vkCmdDraw(graphicsCmdBuffers[i], (uint32_t)vertices.size(), 1, 0, 0);
			vkCmdDrawIndexed(graphicsCmdBuffers[i], (uint32_t)indices.size(), 1, 0, 0, 0);

			vkCmdEndRenderPass(graphicsCmdBuffers[i]);

			if (vkEndCommandBuffer(graphicsCmdBuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Error in ending command buffer recording");
			}
		}

		// Final frame rendering command buffer
		mvCmdBuffers.resize(swapChainframeBuffers.size());

		if (vkAllocateCommandBuffers(logicalDevice, &cmdBufferAllocInfo, mvCmdBuffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allocate command buffers");
		}

		for (int i = 0; i < mvCmdBuffers.size(); i++)
		{
			VkCommandBufferBeginInfo cmdBuffBeginInfo = {};
			cmdBuffBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			cmdBuffBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
			cmdBuffBeginInfo.pInheritanceInfo = nullptr;

			if (vkBeginCommandBuffer(mvCmdBuffers[i], &cmdBuffBeginInfo) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to begin command buffer");
			}

			VkRenderPassBeginInfo renderPassBeginInfo = {};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.renderPass = renderPass;
			renderPassBeginInfo.framebuffer = swapChainframeBuffers[i];
			renderPassBeginInfo.renderArea.offset = { 0,0 };
			renderPassBeginInfo.renderArea.extent = imageExtend;

			std::array<VkClearValue, 3> clearVals = {};
			clearVals[0].color = { 0.0f, 0.0f, 0.0f, 1.f };
			clearVals[1].depthStencil = { 1.0f,0 };
			clearVals[2].color = { 0.0f, 0.0f, 0.0f, 1.f };

			renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearVals.size());
			renderPassBeginInfo.pClearValues = clearVals.data();

			vkCmdBeginRenderPass(mvCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = {};
			viewport.x = viewport.y = 0;
			viewport.width = imageExtend.width/2.0f;
			viewport.height = (float)imageExtend.height;
			viewport.maxDepth = 1;
			viewport.minDepth = 0;

			VkRect2D scissorRect = {};
			scissorRect.extent = { imageExtend.width/2 ,imageExtend.height};
			scissorRect.offset = { 0,0 };

			vkCmdSetViewport(mvCmdBuffers[i], 0, 1, &viewport);
			vkCmdSetScissor(mvCmdBuffers[i], 0, 1, &scissorRect);

			vkCmdBindDescriptorSets(mvCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, mvPipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

			vkCmdBindPipeline(mvCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, mvFramePipelines[0]);
			vkCmdDraw(mvCmdBuffers[i], 4, 1, 0, 0);

			viewport.x = imageExtend.width / 2.0f;
			scissorRect.offset.x = imageExtend.width / 2;
			vkCmdSetViewport(mvCmdBuffers[i], 0, 1, &viewport);
			vkCmdSetScissor(mvCmdBuffers[i], 0, 1, &scissorRect);
			vkCmdBindPipeline(mvCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, mvFramePipelines[1]);
			vkCmdDraw(mvCmdBuffers[i], 4, 1, 0, 0);

			vkCmdEndRenderPass(mvCmdBuffers[i]);

			if (vkEndCommandBuffer(mvCmdBuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Error in ending command buffer recording");
			}
		}
	}

	void drawFrame()
	{
		vkWaitForFences(logicalDevice, 1, &fences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

		uint32_t swapChainIdx;
		VkResult result = vkAcquireNextImageKHR(logicalDevice, swapChain, std::numeric_limits<uint64_t>::max(),
			imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &swapChainIdx);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			std::cout << "Swap chain and pipeline is out dated,Recreating them!" << std::endl;
			recreateSwapchain();
			return;
		}
		else if (result == VK_SUBOPTIMAL_KHR)
		{
			std::cout << "Swap chain is not optimal still usable";
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to acquire image from swap chain to submit render command to graphics queue");
		}

		updateProjectionData(swapChainIdx);

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkSemaphore signalSemaphores[] = { imageRenderedSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

		VkSubmitInfo cmdBufferSubmitInfo = {};
		cmdBufferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		cmdBufferSubmitInfo.commandBufferCount = 1;
		cmdBufferSubmitInfo.pCommandBuffers = &graphicsCmdBuffers[swapChainIdx];
		cmdBufferSubmitInfo.waitSemaphoreCount = 1;
		cmdBufferSubmitInfo.pWaitSemaphores = waitSemaphores;
		cmdBufferSubmitInfo.pWaitDstStageMask = waitStages;
		cmdBufferSubmitInfo.signalSemaphoreCount = 1;
		cmdBufferSubmitInfo.pSignalSemaphores = &mvRenderingSemaphore;

		vkWaitForFences(logicalDevice, 1, &mvTaskFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
		vkResetFences(logicalDevice, 1, &mvTaskFence);

		if (vkQueueSubmit(graphicsQueue, 1, &cmdBufferSubmitInfo, mvTaskFence) != VK_SUCCESS)
		{
			throw std::runtime_error("Error when submitting command to the queue");
		}

		cmdBufferSubmitInfo.pCommandBuffers = &mvCmdBuffers[swapChainIdx];
		cmdBufferSubmitInfo.pWaitSemaphores = &mvRenderingSemaphore;
		cmdBufferSubmitInfo.pSignalSemaphores = signalSemaphores;

		vkResetFences(logicalDevice, 1, &fences[currentFrame]);

		if (vkQueueSubmit(graphicsQueue, 1, &cmdBufferSubmitInfo, fences[currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("Error when submitting command to the queue");
		}

		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pResults = nullptr;
		presentInfo.pImageIndices = &swapChainIdx;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || bIsWindowResized)
		{
			std::cout << "Swap chain and pipeline is out dated,Recreating them!" << std::endl;
			bIsWindowResized = false;
			recreateSwapchain();
			return;
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to present swap chain to presentation queue");
		}

		currentFrame = (currentFrame + 1) % MAX_PARALLEL_FRAMES;
	}

	void updateProjectionData(uint32_t imageIndex)
	{
		ProjectionData projectionData = getProjectionData();

		void *dataPtr;

		vkMapMemory(logicalDevice, uniformBuffersMemory[imageIndex], 0, sizeof(projectionData), 0, &dataPtr);
		memcpy(dataPtr, &projectionData, sizeof(projectionData));
		vkUnmapMemory(logicalDevice, uniformBuffersMemory[imageIndex]);
	}

	void recreateSwapchain()
	{
		int width = 0, height = 0;

		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(logicalDevice);

		preRecreateSwapchain();

		createSwapChain();
		obtainImageAndImgViews();
		createRenderPass();
		createRenderPipeline();
		createImageResources();
		createDepthResources();
		createFramebuffers();
		allocDescriptorSets();
		allocAndRecordCmdBuffers();
	}

	void preRecreateSwapchain()
	{
		descriptorSets.push_back(textureDescriptorSet);
		vkFreeDescriptorSets(logicalDevice, descriptorPool, (uint32_t)descriptorSets.size(), descriptorSets.data());
		descriptorSets.clear();
		textureDescriptorSet =nullptr;

		cleanFrameBuffers(logicalDevice);
		cleanDepthResource();
		cleanImageResources();
		vkDestroyPipeline(logicalDevice, pipeLine, nullptr);
		for (uint32_t i = 0; i < noOfViews; i++)
		{
			vkDestroyPipeline(logicalDevice, mvFramePipelines[i], nullptr);
		}
		vkDestroyPipelineLayout(logicalDevice, mvPipelineLayout, nullptr);
		vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
		vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
		vkDestroyRenderPass(logicalDevice, mvRenderPass, nullptr);
		cleanImageViews();
		vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
	}

	void createSemaphores()
	{
		VkSemaphoreCreateInfo semaphoreCreateInfo = {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		imageAvailableSemaphores.resize(MAX_PARALLEL_FRAMES);
		imageRenderedSemaphores.resize(MAX_PARALLEL_FRAMES);
		fences.resize(MAX_PARALLEL_FRAMES);

		for (int i = 0; i < MAX_PARALLEL_FRAMES; i++)
		{

			if (vkCreateSemaphore(logicalDevice, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(logicalDevice, &semaphoreCreateInfo, nullptr, &imageRenderedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &fences[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create semaphores for synchronizing");
			}
		}

		vkCreateSemaphore(logicalDevice, &semaphoreCreateInfo, nullptr, &mvRenderingSemaphore);
		vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &mvTaskFence);
	}

	void cleanSemaphores()
	{
		for (int i = 0; i < MAX_PARALLEL_FRAMES; i++)
		{
			vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(logicalDevice, imageRenderedSemaphores[i], nullptr);
			vkDestroyFence(logicalDevice, fences[i], nullptr);
		}
		vkDestroySemaphore(logicalDevice, mvRenderingSemaphore, nullptr);
		vkDestroyFence(logicalDevice, mvTaskFence, nullptr);
	}

	void createImageTextureAndView(std::string path,int pushIndex)
	{
		if (textures.size() >= pushIndex)
		{
			std::runtime_error("Textures container size is limited,You are requesting out of bound location in texture container");
		}
		TextureData& data = textures[pushIndex];

		int texWidth, texHeight, texChannels;

		stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

		VkDeviceSize size = texWidth * texHeight * 4;
		data.mipLevelsCount = static_cast<uint32_t>(std::floor(std::log2(std::max(texHeight, texWidth)))) + 1;

		VkFormatProperties formatProps;
		vkGetPhysicalDeviceFormatProperties(vulkanDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProps);

		if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
		{
			data.mipLevelsCount = 1;
			std::cerr << "Cannot create MipMaps by Image Blit as Linear Filtering is not supported by hardware.Using only base texture" << std::endl;
		}

		if (!pixels)
		{
			throw std::runtime_error("Failed loading texture pixels");
		}

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBufferMemory(size, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer, stagingBufferMemory);

		void *dataPtr;

		vkMapMemory(logicalDevice, stagingBufferMemory, 0, size, 0, &dataPtr);
		memcpy(dataPtr, pixels, size);
		vkUnmapMemory(logicalDevice, stagingBufferMemory);

		stbi_image_free(pixels);

		createImageMemory(VK_FORMAT_R8G8B8A8_UNORM, texWidth, texHeight, VK_SAMPLE_COUNT_1_BIT, data.mipLevelsCount, VK_IMAGE_USAGE_TRANSFER_DST_BIT |
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, data.textureImage, data.textureImageMemory);

		transitionImageLayout(data.textureImage, data.mipLevelsCount, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, data.textureImage, texWidth, texHeight);

		generateMipMaps(data.textureImage, data.mipLevelsCount, texWidth, texHeight);

		// Not needed as all the transitions will be taken care by generateMips
		//transitionImageLayout(textureImage, mipLevelsCount, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		//	VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

		// Creating View
		createImageView(data.textureImage, data.mipLevelsCount, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, data.textureImageView);

	}

	void generateMipMaps(VkImage &image, uint32_t mipMapLevels, uint32_t texWidth, uint32_t texHeight)
	{
		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth, mipHeight = texHeight;

		VkCommandBuffer cmdBuffer = startOneTimeCmdBuffer(&graphicsCmdPool);

		for (uint32_t i = 1; i < mipMapLevels; i++)
		{
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
				nullptr, 0, nullptr, 1, &barrier);

			VkImageBlit blit = {};
			blit.srcOffsets[0] = { 0,0,0 };
			blit.srcOffsets[1] = { mipWidth,mipHeight,1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.srcSubresource.mipLevel = i - 1;
			blit.dstOffsets[0] = { 0,0,0 };
			blit.dstOffsets[1] = { mipWidth>1 ? mipWidth / 2 : 1,mipHeight>1 ? mipHeight / 2 : 1,1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;
			blit.dstSubresource.mipLevel = i;

			vkCmdBlitImage(cmdBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit, VK_FILTER_LINEAR);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
				nullptr, 0, nullptr, 1, &barrier);

			if (mipWidth > 1)
				mipWidth /= 2;
			if (mipHeight > 1)
				mipHeight /= 2;
		}

		// Final mipmap transition to shader read layout
		barrier.subresourceRange.baseMipLevel = mipMapLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
			nullptr, 0, nullptr, 1, &barrier);

		endOneTimeCmdBuffer(cmdBuffer, &graphicsCmdPool);
	}

	void createImageResources()
	{
		VkFormat imageFormat = choosenSurfaceFormat.format;
		createImageMemory(imageFormat, imageExtend.width, imageExtend.height, msaaSampleBitsCount, 1,
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			msaaColorRenderTarget, colorRenderTargetMemory);

		createImageView(msaaColorRenderTarget, 1, imageFormat, VK_IMAGE_ASPECT_COLOR_BIT, colorRenderTargetImageView);

		transitionImageLayout(msaaColorRenderTarget, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		// Multiview image resource
		createImageMemory(imageFormat, imageExtend.width, imageExtend.height, VK_SAMPLE_COUNT_1_BIT, 1, 
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mvColorTexture, mvColorTextureMemory, noOfViews);

		createImageView(mvColorTexture, 1, imageFormat, VK_IMAGE_ASPECT_COLOR_BIT, mvColorTextureImageView,noOfViews,VK_IMAGE_VIEW_TYPE_2D_ARRAY);

		transitionImageLayout(mvColorTexture, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, noOfViews);
	}

	void cleanImageResources()
	{
		vkDestroyImageView(logicalDevice, colorRenderTargetImageView, nullptr);
		vkDestroyImage(logicalDevice, msaaColorRenderTarget, nullptr);
		vkFreeMemory(logicalDevice, colorRenderTargetMemory, nullptr);

		vkDestroyImageView(logicalDevice, mvColorTextureImageView, nullptr);
		vkDestroyImage(logicalDevice, mvColorTexture, nullptr);
		vkFreeMemory(logicalDevice, mvColorTextureMemory, nullptr);
	}

	void createDepthResources()
	{
		depthFormat = chooseDepthImageFormat();
		createImageMemory(depthFormat, imageExtend.width, imageExtend.height, msaaSampleBitsCount, 1,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthTexture, depthTextureMemory);
		VkImageAspectFlags flags = hasStencilFormat(depthFormat) ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
		createImageView(depthTexture, 1, depthFormat, flags, depthTextureImageView);
		transitionImageLayout(depthTexture, 1, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

		createImageMemory(depthFormat, imageExtend.width, imageExtend.height, VK_SAMPLE_COUNT_1_BIT, 1,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mvDepthTexture, mvDepthTextureMemory, noOfViews);
		createImageView(mvDepthTexture, 1, depthFormat, flags, mvDepthTextureImageView,noOfViews, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
		transitionImageLayout(mvDepthTexture, 1, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,noOfViews);
	}

	void cleanDepthResource()
	{
		vkDestroyImageView(logicalDevice, depthTextureImageView, nullptr);
		vkDestroyImage(logicalDevice, depthTexture, nullptr);
		vkFreeMemory(logicalDevice, depthTextureMemory, nullptr);

		vkDestroyImageView(logicalDevice, mvDepthTextureImageView, nullptr);
		vkDestroyImage(logicalDevice, mvDepthTexture,nullptr);
		vkFreeMemory(logicalDevice, mvDepthTextureMemory, nullptr);
	}

	void createTextureSampler()
	{
		VkSamplerCreateInfo samplerCreateInfo = {};
		samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

		samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.addressModeU = samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

		samplerCreateInfo.compareEnable = VK_FALSE;
		samplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;

		samplerCreateInfo.anisotropyEnable = VK_TRUE;
		samplerCreateInfo.maxAnisotropy = 16;

		samplerCreateInfo.mipLodBias = 0;
		samplerCreateInfo.minLod = 0;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		for (TextureData& td : textures)
		{
			samplerCreateInfo.maxLod = static_cast<float>(td.mipLevelsCount);
			if (vkCreateSampler(logicalDevice, &samplerCreateInfo, nullptr, &td.textureSampler) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create texture sampler");
			}
		}

		samplerCreateInfo.addressModeU = samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

		samplerCreateInfo.compareEnable = VK_FALSE;
		samplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;

		samplerCreateInfo.anisotropyEnable = VK_TRUE;
		samplerCreateInfo.maxAnisotropy = 16;

		samplerCreateInfo.mipLodBias = 0;
		samplerCreateInfo.minLod = 0;
		samplerCreateInfo.maxLod = 1;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		if (vkCreateSampler(logicalDevice, &samplerCreateInfo, nullptr, &mvColorTextureSampler) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create texture sampler for multiview render target");
		}
	}

	void createImageView(VkImage image, uint32_t mipLevels, VkFormat format, VkImageAspectFlags aspectFlags, VkImageView &imageView, 
		uint32_t arrayLayers = 1, VkImageViewType imageType= VK_IMAGE_VIEW_TYPE_2D)
	{
		VkImageViewCreateInfo imgViewCreateInfo = {};
		imgViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imgViewCreateInfo.image = image;

		imgViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imgViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imgViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imgViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

		imgViewCreateInfo.viewType = imageType;
		imgViewCreateInfo.format = format;

		imgViewCreateInfo.subresourceRange.baseArrayLayer = 0;
		imgViewCreateInfo.subresourceRange.layerCount = arrayLayers;
		imgViewCreateInfo.subresourceRange.baseMipLevel = 0;
		imgViewCreateInfo.subresourceRange.levelCount = mipLevels;
		imgViewCreateInfo.subresourceRange.aspectMask = aspectFlags;

		if (vkCreateImageView(logicalDevice, &imgViewCreateInfo, nullptr, &imageView) != VK_SUCCESS)
		{
			throw std::runtime_error("Creation of image view failed");
		}
	}

	void createBufferMemory(VkDeviceSize size, VkMemoryPropertyFlags memoryProperties, VkBufferUsageFlags usage
		, VkBuffer &buffer, VkDeviceMemory &bufferMemory)
	{

		VkBufferCreateInfo bufferCreateInfo = {};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

		bufferCreateInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
		QueueFamilyIndices selectedQueueIdxs = findQueueFamilyIndices(vulkanDevice);

		if (selectedQueueIdxs.graphicsCmdQueue != selectedQueueIdxs.transferQueue)
		{
			uint32_t queueIndices[] = { (uint32_t)selectedQueueIdxs.graphicsCmdQueue,(uint32_t)selectedQueueIdxs.transferQueue };
			bufferCreateInfo.queueFamilyIndexCount = 2;
			bufferCreateInfo.pQueueFamilyIndices = queueIndices;
		}
		else
		{
			bufferCreateInfo.queueFamilyIndexCount = 0;
			bufferCreateInfo.pQueueFamilyIndices = nullptr;
		}

		bufferCreateInfo.usage = usage;
		bufferCreateInfo.size = size;

		if (vkCreateBuffer(logicalDevice, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed creating vertex buffer");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

		VkMemoryAllocateInfo memAllocInfo = {};
		memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllocInfo.allocationSize = memRequirements.size;
		memAllocInfo.memoryTypeIndex = chooseMemoryType(memRequirements.memoryTypeBits, memoryProperties);

		if (vkAllocateMemory(logicalDevice, &memAllocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed allocating memory from physical device to vertex buffer");
		}

		vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0);
	}

	void copyBuffer(VkBuffer &srcBuffer, VkBuffer &dstBuffer, VkDeviceSize size)
	{

		VkCommandBuffer copyCmdBuffer = startOneTimeCmdBuffer();

		VkBufferCopy copyInfo = {};
		copyInfo.dstOffset = 0;
		copyInfo.srcOffset = 0;
		copyInfo.size = size;

		vkCmdCopyBuffer(copyCmdBuffer, srcBuffer, dstBuffer, 1, &copyInfo);

		endOneTimeCmdBuffer(copyCmdBuffer);
	}

	void createImageMemory(VkFormat imageFormat, int imageWidth, int imageHeight, VkSampleCountFlagBits sampleCountFlagBits, uint32_t mipLevels, VkImageUsageFlags usageFlags,
		VkMemoryPropertyFlags imageProperties, VkImage &image, VkDeviceMemory &imageMemory,uint32_t arrayLayers=1)
	{
		VkImageCreateInfo imageCreateInfo = {};
		imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

		imageCreateInfo.format = imageFormat;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.arrayLayers = arrayLayers;
		imageCreateInfo.extent.depth = 1;
		imageCreateInfo.extent.width = (uint32_t)imageWidth;
		imageCreateInfo.extent.height = (uint32_t)imageHeight;
		imageCreateInfo.mipLevels = mipLevels;
		imageCreateInfo.tiling = (imageProperties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? VK_IMAGE_TILING_OPTIMAL : VK_IMAGE_TILING_LINEAR;
		imageCreateInfo.usage = usageFlags;
		imageCreateInfo.samples = sampleCountFlagBits;
		imageCreateInfo.flags = 0;
		if (transferQueue != graphicsQueue && !(usageFlags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
		{
			QueueFamilyIndices selectedQueueIdxs = findQueueFamilyIndices(vulkanDevice);
			uint32_t queueIndices[] = { (uint32_t)selectedQueueIdxs.graphicsCmdQueue,(uint32_t)selectedQueueIdxs.transferQueue };
			imageCreateInfo.queueFamilyIndexCount = 2;
			imageCreateInfo.pQueueFamilyIndices = queueIndices;
			imageCreateInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
		}
		else
		{
			imageCreateInfo.queueFamilyIndexCount = 0;
			imageCreateInfo.pQueueFamilyIndices = nullptr;
			imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		if (vkCreateImage(logicalDevice, &imageCreateInfo, nullptr, &image) != VK_SUCCESS)
		{
			throw std::runtime_error("Failure in creating texture image");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);

		VkMemoryAllocateInfo allocateInfo = {};
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.memoryTypeIndex = chooseMemoryType(memRequirements.memoryTypeBits, imageProperties);
		allocateInfo.allocationSize = memRequirements.size;

		if (vkAllocateMemory(logicalDevice, &allocateInfo, nullptr, &imageMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed allocating device memory to image");
		}

		vkBindImageMemory(logicalDevice, image, imageMemory, 0);
	}

	void copyBufferToImage(VkBuffer &buffer, VkImage &image, uint32_t width, uint32_t height)
	{

		VkCommandBuffer copyCmdBuffer = startOneTimeCmdBuffer();

		VkBufferImageCopy bufferToImage = {};
		bufferToImage.bufferImageHeight = 0;
		bufferToImage.bufferRowLength = 0;
		bufferToImage.bufferOffset = 0;

		bufferToImage.imageExtent = { width,height,1 };
		bufferToImage.imageOffset = { 0,0,0 };
		bufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		bufferToImage.imageSubresource.baseArrayLayer = 0;
		bufferToImage.imageSubresource.layerCount = 1;
		bufferToImage.imageSubresource.mipLevel = 0;

		vkCmdCopyBufferToImage(copyCmdBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferToImage);

		endOneTimeCmdBuffer(copyCmdBuffer);
	}

	void transitionImageLayout(VkImage image, uint32_t mipLevels, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t arrayLayers = 1)
	{

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			VkImageAspectFlags aspectMaskFlags = hasStencilFormat(format) ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
			barrier.subresourceRange.aspectMask = aspectMaskFlags;
		}
		else
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount =  arrayLayers;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;

		VkPipelineStageFlags srcStageFlags = 0, dstStageFlags = 0;

		VkCommandPool *pool = nullptr;
		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			srcStageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStageFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			srcStageFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;
			dstStageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			pool = &graphicsCmdPool;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			srcStageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStageFlags = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			pool = &graphicsCmdPool;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			srcStageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			pool = &graphicsCmdPool;
		}
		else
		{
			throw std::runtime_error("Not a supported image transition requested");
		}

		VkCommandBuffer transitionCmdBuffer = startOneTimeCmdBuffer(pool);

		vkCmdPipelineBarrier(transitionCmdBuffer, srcStageFlags, dstStageFlags, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		endOneTimeCmdBuffer(transitionCmdBuffer, pool);
	}

	VkCommandBuffer startOneTimeCmdBuffer(VkCommandPool* pool = nullptr)
	{
		VkCommandBuffer cmdBuffer;

		VkCommandBufferAllocateInfo allocationInfo = {};
		allocationInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocationInfo.commandBufferCount = 1;
		allocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocationInfo.commandPool = pool ? *pool : transferCmdPool;

		if (vkAllocateCommandBuffers(logicalDevice, &allocationInfo, &cmdBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to Allocate transfer command buffer");
		}

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.pInheritanceInfo = nullptr;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(cmdBuffer, &beginInfo);

		return cmdBuffer;
	}

	void endOneTimeCmdBuffer(VkCommandBuffer &cmdBuffer, VkCommandPool* pool = nullptr)
	{
		vkEndCommandBuffer(cmdBuffer);

		VkSubmitInfo cmdSubmitInfo = {};
		cmdSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		cmdSubmitInfo.commandBufferCount = 1;
		cmdSubmitInfo.pCommandBuffers = &cmdBuffer;

		VkQueue queueToSubmitTo = pool ? graphicsQueue : transferQueue;

		vkQueueSubmit(queueToSubmitTo, 1, &cmdSubmitInfo, nullptr);

		vkQueueWaitIdle(queueToSubmitTo);

		vkFreeCommandBuffers(logicalDevice, pool ? *pool : transferCmdPool, 1, &cmdBuffer);
	}

	VkSampleCountFlagBits getMsaaSampleCount()
	{
		VkPhysicalDeviceProperties deviceProps;
		vkGetPhysicalDeviceProperties(vulkanDevice, &deviceProps);

		VkSampleCountFlags countFlags = std::min(deviceProps.limits.framebufferColorSampleCounts,
			deviceProps.limits.framebufferDepthSampleCounts);

		if (countFlags & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
		if (countFlags & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
		if (countFlags & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
		if (countFlags & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
		if (countFlags & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
		if (countFlags & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;

		return VK_SAMPLE_COUNT_1_BIT;
	}

	VkShaderModule createShaderModule(std::vector<char> shaderCode)
	{
		VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
		shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shaderModuleCreateInfo.codeSize = shaderCode.size();
		shaderModuleCreateInfo.pCode = reinterpret_cast<uint32_t *>(shaderCode.data());

		VkShaderModule shaderModule;

		if (vkCreateShaderModule(logicalDevice, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create shader module from shader code");
		}

		return shaderModule;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT *pMessageData,
		void* pUserData
	)
	{
		std::cerr << pMessageData->pMessage << std::endl;
		return VK_FALSE;
	}

	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		RenderingApplication* app = reinterpret_cast<RenderingApplication*>(glfwGetWindowUserPointer(window));
		if (key == GLFW_KEY_W && action == GLFW_RELEASE)
		{
			// Increase distortion value by 0.1;
			app->currentDistAlpha += 0.1f;
			app->currentDistAlpha = glm::clamp(app->currentDistAlpha, -1.f, 1.f);
		}
		else if (key == GLFW_KEY_S && action == GLFW_RELEASE)
		{
			// descreases distortion value by 0.1;
			app->currentDistAlpha -= 0.1f;
			app->currentDistAlpha = glm::clamp(app->currentDistAlpha, -1.f, 1.f);
		}
		if (key == GLFW_KEY_T && action == GLFW_RELEASE)
		{
			// Toggles distortionAlpha
			app->currentDistAlpha = app->currentDistAlpha == 0.0f ? app->defaultDistortionAlpha : 0.0f;
		}
	}

	std::vector<char> readShaderFile(const std::string &fileName)
	{
		std::ifstream file(fileName, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("Shader loading failed");
		}

		size_t count = (size_t)file.tellg();

		std::vector<char> fileData(count);
		file.seekg(0);
		file.read(fileData.data(), count);
		file.close();
		return fileData;
	}
	
	ProjectionData getProjectionData()
	{
		static auto initTime = std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();

		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - initTime).count();


		ProjectionData data;
		data.modelTransform = glm::rotate(glm::mat4(1.0f), glm::radians(270.0f)/* *time*/, glm::vec3(0.0f, 0.0f, 1.0f));
		data.modelTransform = glm::rotate(data.modelTransform, glm::radians(180.f), glm::vec3(1.0f, 0.0f, 0.0f));

		float halfEyeSeperation = 0.5f*eyeSeperation;

		// Calculating from resolution
		float aspectRatio = (imageExtend.width*0.5f) / imageExtend.height;

		glm::vec3 right = glm::cross(glm::vec3(0, 0, 1),(glm::vec3(0, 0, 0)- cameraPos));

		// Left eye

		data.projectionTransforms[0] = glm::perspective(glm::radians(fovY), aspectRatio, nearClip, farClip);
		data.projectionTransforms[0][1][1] *= -1;// since GLM is for OpenGL and Y clip coordinate is inverted in OpenGL

		data.viewTransforms[0] = glm::lookAt(cameraPos - (right*halfEyeSeperation), glm::vec3(0, 0, 0) - (right*halfEyeSeperation), glm::vec3(0, 0, 1));

		// Right eye
		
		data.projectionTransforms[1] = glm::perspective(glm::radians(fovY), aspectRatio, nearClip,farClip);
		data.projectionTransforms[1][1][1] *= -1;

		data.viewTransforms[1] = glm::lookAt(cameraPos + (right*halfEyeSeperation), glm::vec3(0, 0, 0) + (right*halfEyeSeperation), glm::vec3(0, 0, 1));

		data.distortionAlpha = currentDistAlpha;
		data.timeSinceStart = time;
		return data;
	}

	void loadModel(std::string path)
	{
		tinyobj::attrib_t attribs;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> mtls;

		std::string err;

		if (!tinyobj::LoadObj(&attribs, &shapes, &mtls, &err, path.c_str()))
		{
			throw std::runtime_error(err);
		}

		std::unordered_map<Vertex, uint32_t> uniqueVertexIndexMap;

		for (tinyobj::shape_t &shape : shapes)
		{
			for (tinyobj::index_t &index : shape.mesh.indices)
			{
				Vertex vert = {};

				vert.textureCoord = {
					attribs.texcoords[2 * index.texcoord_index + 0],
					1.0f - attribs.texcoords[2 * index.texcoord_index + 1]
				};

				vert.position = {
					attribs.vertices[3 * index.vertex_index + 0],
					attribs.vertices[3 * index.vertex_index + 1],
					attribs.vertices[3 * index.vertex_index + 2]
				};

				vert.color = { 1.0f,1.0f ,1.0f };

				if (uniqueVertexIndexMap.find(vert) == uniqueVertexIndexMap.end())
				{
					uniqueVertexIndexMap[vert] = (uint32_t)vertices.size();
					vertices.push_back(vert);
				}

				indices.push_back(uniqueVertexIndexMap[vert]);
			}
		}

		std::cout << "Number of vertices : " << vertices.size() << std::endl;
	}

	void createCylinder(float height,float radius,int numberOfSlices,float angle)
	{
		std::unordered_map<Vertex, uint32_t> uniqueVertexIndexMap;

		float myAngle = glm::clamp(angle, 0.0f, 360.0f);

		float startAngle = -(myAngle *0.5f);
		float halfHeight = height * 0.5f;

		Vertex middleT = {};
		middleT.position= glm::vec3(0, 0, halfHeight);
		middleT.textureCoord = { 0,1 };
		uint32_t mT = 0;
		uniqueVertexIndexMap[middleT] = mT;
		vertices.push_back(middleT);
		Vertex middleB = {};
		middleB.position =-1.0f * middleT.position;
		middleB.textureCoord = { 0,0 };
		uint32_t mB = 1;
		uniqueVertexIndexMap[middleB] = mB;
		vertices.push_back(middleB);

		Vertex firstT = {};
		firstT.position = glm::vec3(radius * cos(glm::radians(startAngle)), radius * sin(glm::radians(startAngle)), halfHeight);
		firstT.textureCoord = { 0,1 };
		uint32_t fT=0;
		Vertex firstB = {};
		firstB.position = glm::vec3(firstT.position.x, firstT.position.y, -halfHeight);
		firstB.textureCoord = { 0,0 };
		uint32_t fB=0;

		for (int i = 0; i < numberOfSlices; i++)
		{
			float ratio = (i + 1) / (float)numberOfSlices;
			float cAngle = startAngle + (myAngle * ratio);

			Vertex secondT = {}; 
			secondT.position = glm::vec3(radius * cos(glm::radians(cAngle)), radius * sin(glm::radians(cAngle)), halfHeight);
			secondT.textureCoord = { ratio,1 };
			uint32_t sT;

			Vertex secondB = {};
			secondB.position= glm::vec3(secondT.position.x, secondT.position.y, -halfHeight);
			secondB.textureCoord = { ratio,0 };
			uint32_t sB;

			// First Triangle
			if (uniqueVertexIndexMap.find(secondT) == uniqueVertexIndexMap.end())
			{
				uniqueVertexIndexMap[secondT] = sT = (uint32_t)vertices.size();
				vertices.push_back(secondT);
			}
			indices.push_back(uniqueVertexIndexMap[secondT]);
			if (vertices[fB] != firstB && uniqueVertexIndexMap.find(firstB) == uniqueVertexIndexMap.end())
			{
				uniqueVertexIndexMap[firstB] = fB =(uint32_t)vertices.size();
				vertices.push_back(firstB);
			}
			indices.push_back(fB);
			if (vertices[fT] != firstT && uniqueVertexIndexMap.find(firstT) == uniqueVertexIndexMap.end())
			{
				uniqueVertexIndexMap[firstT] = fT = (uint32_t)vertices.size();
				vertices.push_back(firstT);
			}
			indices.push_back(fT);

			// Second Triangle
			indices.push_back(sT);
			if (uniqueVertexIndexMap.find(secondB) == uniqueVertexIndexMap.end())
			{
				uniqueVertexIndexMap[secondB] = sB = (uint32_t)vertices.size();
				vertices.push_back(secondB);
			}
			indices.push_back(uniqueVertexIndexMap[secondB]);
			indices.push_back(fB);
			
			// Top Triangle
			indices.push_back(mT);
			indices.push_back(sT);
			indices.push_back(fT);


			// Bottom Triangle
			indices.push_back(mB);
			indices.push_back(fB);
			indices.push_back(sB);

			firstT = secondT;
			fT = sT;
			firstB = secondB;
			fB = sB;
		}
		
		std::cout << "Number of vertices : " << vertices.size() << std::endl;
	}


	const std::string MDL_PATH = "Models/earth.obj", TEXTURE_PATH = "Textures/earth.jpg";


	glm::vec3 cameraPos = { 1,400,1 };
	float eyeSeperation = 0.064f;
	float nearClip = 0.1f;
	float farClip = 10000.f;
	float fovY = 80.f;
	float cylinderH = 600;
	float cylinderR = 300;
	float cylinderAngle =180;
	int noOfSlices = 45;
};



int main()
{
	RenderingApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}