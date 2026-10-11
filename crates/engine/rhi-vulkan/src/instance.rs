//! 实例层：loader 加载、扩展/层校验、[`ash::Instance`] 生命周期与窗口表面创建。
//!
//! 设计取舍：**任何失败都返回 [`RhiError`]，绝不 panic**。没有 Vulkan 的环境
//! （缺 loader、只有 loader 没有 ICD 驱动、驱动不支持 1.3）都让上层拿到可诊断的错误并降级，
//! 这样 `dedicated-server` 与 `tests/integration` 在无 GPU 的 CI 里照常构建与测试。
//!
//! 窗口表面（surface）走 `ash-window`：把「平台相关的句柄 → `VkSurfaceKHR`」这段
//! 唯一需要 `unsafe` 的胶水收敛在本模块。

use crate::{version_string, RhiError, REQUIRED_API_VERSION};
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::CStr;

/// 应用名（NUL 结尾的静态字节串，供 `VkApplicationInfo` 使用）。
const APPLICATION_NAME: &[u8] = b"TrueHorizon\0";
/// 引擎名。
const ENGINE_NAME: &[u8] = b"TrueHorizon Engine\0";
/// 校验层名（仅在显式要求校验时才作为硬性依赖检查）。
const VALIDATION_LAYER: &[u8] = b"VK_LAYER_KHRONOS_validation\0";
/// 校验层的可读名字（用于与驱动的层列表比较）。
const VALIDATION_LAYER_NAME: &str = "VK_LAYER_KHRONOS_validation";

/// Vulkan loader（运行时动态加载 `vulkan-1.dll` / `libvulkan.so.1`）。
///
/// 加载失败是**正常返回路径**（无 Vulkan 的容器 / CI），不是错误用法。
#[derive(Clone)]
pub struct VulkanLoader {
    entry: ash::Entry,
}

impl VulkanLoader {
    /// 加载系统 Vulkan loader。
    pub fn load() -> Result<Self, RhiError> {
        // SAFETY: `Entry::load` 只是 dlopen 系统 loader 并取函数指针；
        // 之后所有调用都通过本类型，且 loader 的生命周期由 `ash::Entry` 持有。
        unsafe { ash::Entry::load() }
            .map(|entry| Self { entry })
            .map_err(|error| RhiError::NoVulkanImplementation {
                reason: error.to_string(),
            })
    }

    /// 底层 `ash` 入口（供本 crate 的其余模块使用）。
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// loader 支持的**实例**版本；`load` 缺失 `vkEnumerateInstanceVersion` 时按 Vulkan 1.0 处理。
    pub fn instance_api_version(&self) -> Result<u32, RhiError> {
        // SAFETY: 仅查询 loader 版本，不创建任何对象。
        let version = unsafe { self.entry.try_enumerate_instance_version() }.map_err(|result| {
            RhiError::InstanceCreationFailed(format!("查询 loader 版本失败: {result:?}"))
        })?;
        Ok(version.unwrap_or_else(|| vk::make_api_version(0, 1, 0, 0)))
    }

    /// 驱动报告的可用实例层名字。
    pub fn available_layers(&self) -> Result<Vec<String>, RhiError> {
        // SAFETY: 仅枚举 loader 已知的层。
        let layers =
            unsafe { self.entry.enumerate_instance_layer_properties() }.map_err(|result| {
                RhiError::InstanceCreationFailed(format!("枚举实例层失败: {result:?}"))
            })?;
        Ok(layers
            .iter()
            .filter_map(|layer| layer.layer_name_as_c_str().ok())
            .map(|name| name.to_string_lossy().into_owned())
            .collect())
    }

    /// 驱动报告的可用实例扩展名字。
    pub fn available_extensions(&self) -> Result<Vec<String>, RhiError> {
        Ok(self
            .instance_extensions()?
            .iter()
            .filter_map(|extension| extension.extension_name_as_c_str().ok())
            .map(|name| name.to_string_lossy().into_owned())
            .collect())
    }

    /// 当前环境是否提供校验层。
    pub fn validation_layer_available(&self) -> bool {
        self.available_layers()
            .map(|layers| layers.iter().any(|layer| layer == VALIDATION_LAYER_NAME))
            .unwrap_or(false)
    }

    /// 原始扩展属性列表（避免二次枚举）。
    fn instance_extensions(&self) -> Result<Vec<vk::ExtensionProperties>, RhiError> {
        // SAFETY: 仅枚举 loader 已知的实例扩展。
        unsafe { self.entry.enumerate_instance_extension_properties(None) }.map_err(|result| {
            RhiError::InstanceCreationFailed(format!("枚举实例扩展失败: {result:?}"))
        })
    }

    /// 按配置创建实例；要求的扩展/层缺失时返回明确错误而不是让驱动静默忽略。
    pub fn create_instance(&self, config: &InstanceConfig) -> Result<VulkanInstance, RhiError> {
        let available = self.instance_extensions()?;
        for required in &config.extensions {
            if !available.iter().any(|extension| {
                extension
                    .extension_name_as_c_str()
                    .map(|name| name == *required)
                    .unwrap_or(false)
            }) {
                return Err(RhiError::ExtensionMissing(
                    required.to_str().unwrap_or("(非 UTF-8 扩展名)"),
                ));
            }
        }
        if config.enable_validation && !self.validation_layer_available() {
            return Err(RhiError::LayerMissing(VALIDATION_LAYER_NAME));
        }

        let application_name = CStr::from_bytes_with_nul(APPLICATION_NAME).expect("静态 NUL 结尾");
        let engine_name = CStr::from_bytes_with_nul(ENGINE_NAME).expect("静态 NUL 结尾");
        let mut extension_pointers: Vec<*const std::os::raw::c_char> =
            config.extensions.iter().map(|ext| ext.as_ptr()).collect();
        let mut layer_pointers: Vec<*const std::os::raw::c_char> = if config.enable_validation {
            vec![CStr::from_bytes_with_nul(VALIDATION_LAYER)
                .expect("静态 NUL 结尾")
                .as_ptr()]
        } else {
            Vec::new()
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(application_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(config.api_version);
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_pointers)
            .enabled_extension_names(&extension_pointers);

        // SAFETY: `create_info` 引用的应用信息、扩展名、层名都在本函数内保持存活，
        // 满足 `vkCreateInstance` 对指针生命周期的要求；成功后实例由返回值独占持有。
        let instance =
            unsafe { self.entry.create_instance(&create_info, None) }.map_err(|result| {
                RhiError::InstanceCreationFailed(format!("vkCreateInstance 失败: {result:?}"))
            })?;

        // 指针向量只服务于创建调用，稍后释放不再被引用。
        extension_pointers.clear();
        layer_pointers.clear();

        Ok(VulkanInstance {
            entry: self.entry.clone(),
            instance,
            api_version: config.api_version,
        })
    }
}

/// 实例创建配置。
#[derive(Clone, Debug)]
pub struct InstanceConfig {
    api_version: u32,
    enable_validation: bool,
    extensions: Vec<&'static CStr>,
}

impl InstanceConfig {
    /// 无窗口（无头）配置：不要求任何表面扩展。
    pub fn headless() -> Self {
        Self {
            api_version: REQUIRED_API_VERSION,
            enable_validation: false,
            extensions: Vec::new(),
        }
    }

    /// 覆盖请求的实例 API 版本。
    pub fn with_api_version(mut self, api_version: u32) -> Self {
        self.api_version = api_version;
        self
    }

    /// 是否启用 `VK_LAYER_KHRONOS_validation`（层缺失时创建实例会失败）。
    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.enable_validation = enabled;
        self
    }

    /// 追加一个必需实例扩展。
    pub fn require_extension(mut self, extension: &'static CStr) -> Self {
        if !self.extensions.contains(&extension) {
            self.extensions.push(extension);
        }
        self
    }

    /// 按显示句柄追加平台表面所需的实例扩展（`VK_KHR_surface` + 平台扩展）。
    pub fn with_surface(mut self, display_handle: RawDisplayHandle) -> Result<Self, RhiError> {
        let required = ash_window::enumerate_required_extensions(display_handle)
            .map_err(|result| RhiError::SurfaceUnsupported(format!("{result:?}")))?;
        for pointer in required {
            // SAFETY: `enumerate_required_extensions` 返回的是指向静态 NUL 结尾 C 字符串的指针
            // （ash-window 内部为 `const` 数组，元素取自 `ash::khr::*::NAME` 之类的 `&'static CStr`）。
            let extension: &'static CStr = unsafe { CStr::from_ptr(*pointer) };
            if !self.extensions.contains(&extension) {
                self.extensions.push(extension);
            }
        }
        Ok(self)
    }

    /// 请求的实例 API 版本。
    pub fn api_version(&self) -> u32 {
        self.api_version
    }

    /// 是否要求校验层。
    pub fn validation_enabled(&self) -> bool {
        self.enable_validation
    }

    /// 必需实例扩展列表。
    pub fn extensions(&self) -> &[&'static CStr] {
        &self.extensions
    }
}

impl Default for InstanceConfig {
    fn default() -> Self {
        Self::headless()
    }
}

/// 已创建的 Vulkan 实例；析构时自动 `vkDestroyInstance`。
pub struct VulkanInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    api_version: u32,
}

impl VulkanInstance {
    /// `ash` 实例对象。
    pub fn handle(&self) -> &ash::Instance {
        &self.instance
    }

    /// 原始 `VkInstance` 句柄。
    pub fn raw_handle(&self) -> vk::Instance {
        self.instance.handle()
    }

    /// loader 入口（创建 surface 需要）。
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// 请求的实例 API 版本。
    pub fn api_version(&self) -> u32 {
        self.api_version
    }

    /// 枚举物理设备。
    pub fn physical_devices(&self) -> Result<Vec<vk::PhysicalDevice>, RhiError> {
        // SAFETY: 纯查询；实例有效。
        unsafe { self.instance.enumerate_physical_devices() }.map_err(|result| {
            RhiError::NoSuitableDevice {
                missing: vec![match result {
                    vk::Result::ERROR_INITIALIZATION_FAILED => {
                        "无可用 ICD 驱动（vkEnumeratePhysicalDevices 初始化失败）"
                    },
                    _ => "物理设备枚举失败",
                }],
            }
        })
    }

    /// 为窗口/显示句柄创建 `VkSurfaceKHR`。
    ///
    /// 调用方必须保证窗口在 surface 销毁前一直存活，且只能在**拥有 surface 扩展**的实例上调用。
    pub fn create_surface(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<vk::SurfaceKHR, RhiError> {
        // SAFETY: ash-window 要求窗口/显示连接在返回的 surface 之前不被销毁，
        // 该约束由调用方（`VulkanBackend::new_for_window` 的文档）承担。
        unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                display_handle,
                window_handle,
                None,
            )
        }
        .map_err(|result| RhiError::SurfaceCreationFailed(format!("{result:?}")))
    }

    /// 销毁由 [`Self::create_surface`] 创建的 surface。
    pub fn destroy_surface(&self, surface: vk::SurfaceKHR) {
        // SAFETY: surface 由本实例创建且尚未销毁；调用方承诺不再使用该句柄。
        unsafe {
            ash::khr::surface::Instance::new(&self.entry, &self.instance)
                .destroy_surface(surface, None)
        };
    }

    /// 可读的实例版本串。
    pub fn api_version_string(&self) -> String {
        version_string(self.api_version)
    }
}

impl std::fmt::Debug for VulkanInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanInstance")
            .field("api_version", &self.api_version_string())
            .finish_non_exhaustive()
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        // SAFETY: 实例由本类型独占持有；上层已先行销毁设备与 surface（见 `VulkanBackend`）。
        unsafe { self.instance.destroy_instance(None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_loading_never_panics() {
        match VulkanLoader::load() {
            Ok(loader) => {
                let version = loader.instance_api_version().expect("loader 应能报告版本");
                assert!(vk::api_version_major(version) >= 1);
            },
            Err(RhiError::NoVulkanImplementation { reason }) => {
                assert!(!reason.is_empty(), "缺少 loader 时必须给出原因");
            },
            Err(other) => panic!("加载 loader 只应返回 NoVulkanImplementation，实际: {other}"),
        }
    }

    #[test]
    fn required_surface_extensions_follow_the_display_handle() {
        let config = InstanceConfig::headless();
        assert!(config.extensions().is_empty());

        #[cfg(target_os = "windows")]
        let display = RawDisplayHandle::Windows(raw_window_handle::WindowsDisplayHandle::new());
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        let display = RawDisplayHandle::Xlib(raw_window_handle::XlibDisplayHandle::new(None, 0));
        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "freebsd")))]
        let display = RawDisplayHandle::Web(raw_window_handle::WebDisplayHandle::new());

        let config = config.with_surface(display).expect("已知平台应能给出扩展");
        let names: Vec<String> = config
            .extensions()
            .iter()
            .map(|extension| extension.to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().any(|name| name == "VK_KHR_surface"),
            "surface 扩展必须包含 VK_KHR_surface，实际: {names:?}"
        );
    }

    #[test]
    fn validation_request_is_checked_not_ignored() {
        let Ok(loader) = VulkanLoader::load() else {
            return;
        };
        let required = loader.validation_layer_available();
        let config = InstanceConfig::headless().with_validation(true);
        match loader.create_instance(&config) {
            Ok(_) => assert!(required, "校验层不可用时不允许创建成功"),
            Err(RhiError::LayerMissing(name)) => {
                assert_eq!(name, VALIDATION_LAYER_NAME);
                assert!(!required);
            },
            Err(other) => panic!("只应返回 LayerMissing，实际: {other}"),
        }
    }
}
