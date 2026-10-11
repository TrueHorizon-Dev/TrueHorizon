//! Wasm Mod 宿主（`wasmtime` 沙箱）。
//!
//! Mod 是**不可信的第三方代码**，因此：
//!
//! * 默认零权限：读世界、写方块、网络、文件系统都必须由清单显式申请；
//! * 所有宿主能力通过导入函数显式暴露，Mod 拿不到任何裸指针或文件句柄；
//! * ABI 带版本号，版本不符直接拒绝加载（绝不做「尽力兼容」）。
//!
//! 骨架现状：`Engine` 生命周期、清单校验与模块编译已可用（含单元测试），
//! 实例化、燃料计量与宿主函数导出列在路线图上。

use wasmtime::{Engine, Module};

/// 当前 Mod ABI 版本。
pub const MOD_ABI_VERSION: u32 = 1;

/// Mod 申请的权限。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ModPermissions {
    /// 读取世界（采样地形、查询方块）。
    pub read_world: bool,
    /// 写入方块（需服务端权威校验通过）。
    pub write_blocks: bool,
    /// 发起网络请求。
    pub network: bool,
    /// 读写文件系统。
    pub filesystem: bool,
}

/// Mod 清单（随 Mod 一起分发）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModManifest {
    /// 唯一名称（反向域名风格，例如 `dev.truehorizon.weather`）。
    pub name: String,
    /// 语义化版本。
    pub version: String,
    /// 目标 ABI 版本。
    pub abi: u32,
    /// 申请权限。
    pub permissions: ModPermissions,
}

impl ModManifest {
    /// 构造一个零权限清单。
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            abi: MOD_ABI_VERSION,
            permissions: ModPermissions::default(),
        }
    }

    /// ABI 是否被本宿主接受。
    pub fn is_compatible(&self) -> bool {
        self.abi == MOD_ABI_VERSION
    }
}

/// Mod 加载错误。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModError {
    /// ABI 版本不匹配。
    AbiMismatch {
        /// 宿主支持的版本。
        expected: u32,
        /// 清单声明的版本。
        found: u32,
    },
    /// 编译失败（不是合法 Wasm、用了未启用的特性等）。
    Compile(String),
    /// 权限不足。
    Denied {
        /// 被拒绝的能力。
        capability: &'static str,
    },
}

impl core::fmt::Display for ModError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AbiMismatch { expected, found } => {
                write!(f, "Mod ABI 不匹配：宿主 {expected}，Mod {found}")
            }
            Self::Compile(message) => write!(f, "Mod 编译失败: {message}"),
            Self::Denied { capability } => write!(f, "Mod 未获得权限: {capability}"),
        }
    }
}

impl std::error::Error for ModError {}

/// 已编译的 Mod。
pub struct ModModule {
    manifest: ModManifest,
    module: Module,
}

impl ModModule {
    /// 清单。
    #[inline]
    pub fn manifest(&self) -> &ModManifest {
        &self.manifest
    }

    /// 底层模块（供实例化阶段使用）。
    #[inline]
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// 导出的符号名（诊断用）。
    pub fn export_names(&self) -> Vec<String> {
        self.module
            .exports()
            .map(|export| export.name().to_string())
            .collect()
    }
}

impl core::fmt::Debug for ModModule {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ModModule")
            .field("name", &self.manifest.name)
            .field("exports", &self.export_names())
            .finish()
    }
}

/// Mod 宿主：持有一个 `wasmtime::Engine`，负责编译与（后续的）实例化。
pub struct ModHost {
    engine: Engine,
}

impl ModHost {
    /// 创建宿主。
    ///
    /// 使用 `wasmtime` 的默认配置：单线程、无 SIMD 之外的非确定性来源，
    /// 保证同一 Mod + 同一输入在任何机器上产生相同结果。
    pub fn new() -> Self {
        Self {
            engine: Engine::default(),
        }
    }

    /// 底层引擎。
    #[inline]
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// 编译一个 Mod。
    ///
    /// TODO(mods): 实例化时需附加 —— 燃料计量（`consume_fuel`）、
    /// 内存上限（`StoreLimits`）、宿主函数白名单（按 [`ModPermissions`] 生成导入表）。
    pub fn compile(&self, manifest: &ModManifest, wasm: &[u8]) -> Result<ModModule, ModError> {
        if !manifest.is_compatible() {
            return Err(ModError::AbiMismatch {
                expected: MOD_ABI_VERSION,
                found: manifest.abi,
            });
        }
        let module = Module::new(&self.engine, wasm)
            .map_err(|error| ModError::Compile(error.to_string()))?;
        Ok(ModModule {
            manifest: manifest.clone(),
            module,
        })
    }

    /// 检查某项能力是否被允许，否则返回 [`ModError::Denied`]。
    pub fn require(manifest: &ModManifest, capability: &'static str) -> Result<(), ModError> {
        let allowed = match capability {
            "read_world" => manifest.permissions.read_world,
            "write_blocks" => manifest.permissions.write_blocks,
            "network" => manifest.permissions.network,
            "filesystem" => manifest.permissions.filesystem,
            _ => false,
        };
        if allowed {
            Ok(())
        } else {
            Err(ModError::Denied { capability })
        }
    }
}

impl Default for ModHost {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 最小合法 Wasm 模块（只有 8 字节的 magic + version，无任何段）。
    const EMPTY_MODULE: [u8; 8] = [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];

    #[test]
    fn empty_module_compiles() {
        let host = ModHost::new();
        let manifest = ModManifest::new("dev.truehorizon.test", "0.1.0");
        let module = host
            .compile(&manifest, &EMPTY_MODULE)
            .expect("空模块应当可编译");
        assert_eq!(module.manifest().name, "dev.truehorizon.test");
        assert!(module.export_names().is_empty(), "空模块没有任何导出");
        assert!(format!("{module:?}").contains("dev.truehorizon.test"));
    }

    #[test]
    fn abi_mismatch_is_rejected_before_compiling() {
        let host = ModHost::new();
        let mut manifest = ModManifest::new("dev.truehorizon.legacy", "0.0.1");
        manifest.abi = MOD_ABI_VERSION + 1;
        assert_eq!(
            host.compile(&manifest, &EMPTY_MODULE).unwrap_err(),
            ModError::AbiMismatch {
                expected: MOD_ABI_VERSION,
                found: MOD_ABI_VERSION + 1
            }
        );
    }

    #[test]
    fn garbage_input_fails_without_panicking() {
        let host = ModHost::new();
        let manifest = ModManifest::new("dev.truehorizon.bad", "0.1.0");
        let error = host.compile(&manifest, b"not a wasm module").unwrap_err();
        assert!(matches!(error, ModError::Compile(_)));
    }

    #[test]
    fn permissions_default_to_deny() {
        let manifest = ModManifest::new("dev.truehorizon.locked", "0.1.0");
        assert!(ModHost::require(&manifest, "read_world").is_err());
        assert!(ModHost::require(&manifest, "filesystem").is_err());
        assert!(ModHost::require(&manifest, "unknown-capability").is_err());

        let mut sandboxed = manifest.clone();
        sandboxed.permissions.read_world = true;
        assert!(ModHost::require(&sandboxed, "read_world").is_ok());
        assert!(ModHost::require(&sandboxed, "write_blocks").is_err());
        assert!(sandboxed.is_compatible());
    }
}
