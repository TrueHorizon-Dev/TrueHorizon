//! 运行环境探测：给 `world-tool diagnose` 与启动日志一个**不持有任何 GPU 资源**的快照。
//!
//! 探测路径与真实创建路径共用同一套查询代码（[`query_candidate`]、[`select_best_candidate`]），
//! 因此「诊断说支持」与「真的能建后端」不会互相矛盾。

use crate::device::{query_candidate, select_best_candidate, DeviceInfo, FeatureRequirements};
use crate::instance::{InstanceConfig, VulkanLoader};
use crate::version_string;

/// 探测结果。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RhiProbe {
    /// 是否加载到 Vulkan loader。
    pub loader_available: bool,
    /// loader 支持的实例版本（`1.3.280` 形式；无法查询时为 `None`）。
    pub loader_api_version: Option<String>,
    /// 是否成功创建实例。
    pub instance_created: bool,
    /// 枚举到的物理设备数量。
    pub physical_devices: usize,
    /// 满足要求的最佳设备摘要（没有则为 `None`）。
    pub device: Option<DeviceInfo>,
    /// 人类可读的结论或失败原因。
    pub message: String,
}

impl RhiProbe {
    /// 构造一个「在此步骤失败」的结果。
    fn failed(message: impl Into<String>) -> Self {
        Self {
            loader_available: false,
            loader_api_version: None,
            instance_created: false,
            physical_devices: 0,
            device: None,
            message: message.into(),
        }
    }

    /// 是否拿到了可用的 GPU 设备。
    pub fn is_ready(&self) -> bool {
        self.device.is_some()
    }
}

/// 探测当前环境；**绝不 panic**，失败原因写进 [`RhiProbe::message`]。
pub fn probe(requirements: &FeatureRequirements) -> RhiProbe {
    let loader = match VulkanLoader::load() {
        Ok(loader) => loader,
        Err(error) => return RhiProbe::failed(format!("无 Vulkan loader：{error}")),
    };
    let loader_api_version = loader.instance_api_version().ok();
    let instance = match loader.create_instance(&InstanceConfig::headless()) {
        Ok(instance) => instance,
        Err(error) => {
            let mut probe = RhiProbe::failed(format!("实例创建失败：{error}"));
            probe.loader_available = true;
            probe.loader_api_version = loader_api_version.map(version_string);
            return probe;
        },
    };
    let physical_devices = instance.physical_devices().unwrap_or_default();
    let candidates = physical_devices
        .iter()
        .enumerate()
        .filter_map(|(index, physical_device)| {
            query_candidate(&instance, *physical_device, index).ok()
        })
        .collect::<Vec<_>>();
    let device = select_best_candidate(&candidates, requirements)
        .ok()
        .map(|selected| candidates[selected].info.clone());

    let message = match &device {
        Some(info) => format!(
            "使用 {}（Vulkan {}，显存 {:.1} GiB）",
            info.name,
            info.api_version_string(),
            info.vram_bytes as f64 / (1u64 << 30) as f64
        ),
        None if physical_devices.is_empty() => {
            "有实例但枚举不到物理设备（通常是没有安装 ICD 驱动）".to_string()
        },
        None => format!(
            "枚举到 {} 个物理设备，但都不满足要求",
            physical_devices.len()
        ),
    };

    RhiProbe {
        loader_available: true,
        loader_api_version: loader_api_version.map(version_string),
        instance_created: true,
        physical_devices: physical_devices.len(),
        device,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_always_reports_something_actionable() {
        let probe = probe(&FeatureRequirements::minimum());
        assert!(!probe.message.is_empty(), "探测必须给出结论或原因");
        assert_eq!(probe.is_ready(), probe.device.is_some());
        if probe.loader_available {
            assert!(probe.loader_api_version.is_some());
        } else {
            assert!(!probe.instance_created);
            assert_eq!(probe.physical_devices, 0);
        }
        if probe.instance_created {
            assert!(probe.loader_available);
        }
    }
}
