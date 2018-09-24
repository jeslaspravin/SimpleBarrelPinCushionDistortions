#pragma once

#include <vulkan/vulkan_core.h>
#include <vector>
#include <glm/glm.hpp>
#include <array>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/hash.hpp"

namespace vulkan
{
	class VulkanTypes
	{
	public:

		// Debug Utils Messenger Callback creating function symbol pointer
		static PFN_vkCreateDebugUtilsMessengerEXT fnVkCreateDebugUtilsMessengerExt;
		// Debug Utils Messenger Callback destroying function symbol pointer
		static PFN_vkDestroyDebugUtilsMessengerEXT fnVkDestroyDebugUtilsMessengerExt;

		static void setupNecessaryApi(VkInstance vkInstance)
		{
			fnVkCreateDebugUtilsMessengerExt= (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vkInstance,
				"vkCreateDebugUtilsMessengerEXT"); 
			fnVkDestroyDebugUtilsMessengerExt = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vkInstance,
				"vkDestroyDebugUtilsMessengerEXT");
		}
	};
	
	struct QueueFamilyIndices
	{
		int graphicsCmdQueue = -1;

		int presentationCmdQueue = -1;

		int transferQueue = -1;

		bool isComplete()
		{
			return graphicsCmdQueue >= 0 && presentationCmdQueue>=0 && transferQueue>=0;
		}
	};

	struct SwapChainSupport
	{
		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		std::vector<VkSurfaceFormatKHR> surfaceFormats;
		std::vector<VkPresentModeKHR> presentModes;

		VkSurfaceFormatKHR chooseSurfaceFormat()
		{
			// If Undefined then Surface accepts any format
			if (surfaceFormats.size() == 1 && surfaceFormats[0].format == VK_FORMAT_UNDEFINED)
			{
				return { VK_FORMAT_B8G8R8A8_UNORM,VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
			}

			// Check whether SRGB color space and RGBA32 format is available
			for (VkSurfaceFormatKHR &surfaceFormat : surfaceFormats)
			{
				if (surfaceFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && surfaceFormat.format == VK_FORMAT_B8G8R8A8_UNORM)
				{
					return surfaceFormat;
				}
			}

			// If nothing expected is available choose first one
			return surfaceFormats[0];
		}

		VkPresentModeKHR choosePresentMode()
		{
			VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

			for (VkPresentModeKHR &presentMode : presentModes)
			{
				if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
				{
					return presentMode;
				}
				else if (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
					bestMode = presentMode;
			}

			return bestMode;
		}
	};

	struct Vertex
	{
		glm::vec3 position;
		glm::vec3 color = {1.0f,1.0f,1.0f};
		glm::vec2 textureCoord;

		static VkVertexInputBindingDescription getBindingDesc()
		{
			VkVertexInputBindingDescription bindingDesc = {};
			bindingDesc.binding = 0;
			bindingDesc.stride = sizeof(Vertex);
			bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			return bindingDesc;
		}

		static std::array<VkVertexInputAttributeDescription, 3> getAttributeDesc()
		{
			std::array<VkVertexInputAttributeDescription, 3> attributeDesc={};

			attributeDesc[0].binding = 0;
			attributeDesc[0].location = 0;
			attributeDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDesc[0].offset = offsetof(Vertex, position);

			attributeDesc[1].binding = 0;
			attributeDesc[1].location = 1;
			attributeDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeDesc[1].offset = offsetof(Vertex, color);

			attributeDesc[2].binding = 0;
			attributeDesc[2].location = 2;
			attributeDesc[2].format = VK_FORMAT_R32G32_SFLOAT;
			attributeDesc[2].offset = offsetof(Vertex, textureCoord);

			return attributeDesc;
		}

		bool operator==(const Vertex &otherVertex) const
		{
			return position == otherVertex.position && textureCoord == otherVertex.textureCoord && color == otherVertex.color;
		}

		bool operator!=(const Vertex &otherVertex) const
		{
			return !(*this == otherVertex);
		}
	};

	struct ProjectionData
	{
		glm::mat4 modelTransform;
		glm::mat4 viewTransforms[2];
		glm::mat4 projectionTransforms[2];
		float distortionAlpha = 0.8f;
		float timeSinceStart;
	};
}

namespace std
{
	template<> struct hash<vulkan::Vertex>
	{
		size_t operator()(const vulkan::Vertex &vertex) const noexcept
		{
			return (hash<glm::vec3>()(vertex.position) ^ 
				(hash<glm::vec3>()(vertex.color) << 1) >> 1) ^
				(hash<glm::vec2>()(vertex.textureCoord)<<1);
		}
	};
}