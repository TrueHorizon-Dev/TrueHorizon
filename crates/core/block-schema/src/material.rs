//! 材质、渲染分类与物理属性。

/// 材质分类。
///
/// 材质决定渲染侧选用哪张纹理图集与哪套着色分支；[`Material::id`] 是上传给 GPU 的
/// 稳定索引，绘制命令里以整型传递（见 `render-contracts::DrawCall`）。
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
)]
#[repr(u8)]
pub enum Material {
    /// 空气/空腔，不渲染。
    #[default]
    Air = 0,
    /// 岩石。
    Rock = 1,
    /// 土壤。
    Soil = 2,
    /// 沙。
    Sand = 3,
    /// 液态水。
    Water = 4,
    /// 冰。
    Ice = 5,
    /// 金属。
    Metal = 6,
    /// 有机物（植被、木质）。
    Organic = 7,
    /// 玻璃等半透明固体。
    Glass = 8,
}

impl Material {
    /// 全部材质，顺序与判别式一致。
    pub const ALL: [Material; 9] = [
        Material::Air,
        Material::Rock,
        Material::Soil,
        Material::Sand,
        Material::Water,
        Material::Ice,
        Material::Metal,
        Material::Organic,
        Material::Glass,
    ];

    /// GPU 侧材质槽位索引。
    #[inline]
    pub const fn id(self) -> u32 {
        self as u32
    }

    /// 由槽位索引还原。
    #[inline]
    pub const fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Air),
            1 => Some(Self::Rock),
            2 => Some(Self::Soil),
            3 => Some(Self::Sand),
            4 => Some(Self::Water),
            5 => Some(Self::Ice),
            6 => Some(Self::Metal),
            7 => Some(Self::Organic),
            8 => Some(Self::Glass),
            _ => None,
        }
    }

    /// 是否需要走半透明通道（必须在不透明几何之后按深度排序绘制）。
    #[inline]
    pub const fn is_translucent(self) -> bool {
        matches!(self, Self::Water | Self::Glass)
    }

    /// 流体材质（由 `world::simulation` 的规则负责流动）。
    #[inline]
    pub const fn is_fluid(self) -> bool {
        matches!(self, Self::Water)
    }
}

/// 渲染分类，决定网格生成阶段的输出桶。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(u8)]
pub enum RenderClass {
    /// 不产生任何几何。
    #[default]
    Invisible = 0,
    /// 不透明，参与贪心合并与遮挡剔除。
    Opaque = 1,
    /// 镂空（叶片、栅栏），不参与面剔除但参与合并。
    Cutout = 2,
    /// 半透明，单独一桶，最后绘制。
    Translucent = 3,
}

impl RenderClass {
    /// 是否产生几何。
    #[inline]
    pub const fn emits_geometry(self) -> bool {
        !matches!(self, Self::Invisible)
    }
}

/// 方块物理属性。
///
/// 单位：密度 `kg/m³`，摩擦与弹性为无量纲 `[0, 1]`，硬度为莫氏尺度的相对值。
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct BlockPhysics {
    /// 是否参与碰撞、遮挡与视线阻挡。
    pub solid: bool,
    /// 密度（kg/m³），驱动浮力与惯性。
    pub density_kg_m3: f32,
    /// 动摩擦系数。
    pub friction: f32,
    /// 恢复系数（弹性）。
    pub restitution: f32,
    /// 相对硬度，用于挖掘耗时与破碎规则。
    pub hardness: f32,
}

impl BlockPhysics {
    /// 空气：非实体、零密度、无摩擦。
    pub const AIR: Self = Self {
        solid: false,
        density_kg_m3: 0.0,
        friction: 0.0,
        restitution: 0.0,
        hardness: 0.0,
    };

    /// 典型岩石：致密、粗糙、几乎不弹。
    pub const ROCK: Self = Self {
        solid: true,
        density_kg_m3: 2700.0,
        friction: 0.7,
        restitution: 0.05,
        hardness: 3.0,
    };

    /// 水：非实体（可行走于其上由仿真另行处理），用于浮力计算。
    pub const WATER: Self = Self {
        solid: false,
        density_kg_m3: 1000.0,
        friction: 0.0,
        restitution: 0.0,
        hardness: 0.0,
    };
}

impl Default for BlockPhysics {
    fn default() -> Self {
        Self::AIR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_ids_are_stable_and_round_trip() {
        for material in Material::ALL {
            assert_eq!(Material::from_id(material.id()), Some(material));
        }
        assert_eq!(Material::from_id(Material::ALL.len() as u32), None);
        assert_eq!(
            Material::Air.id(),
            0,
            "材质 0 必须恒为空气，与 BlockId::AIR 对齐"
        );
    }

    #[test]
    fn translucent_and_fluid_partition_is_sane() {
        assert!(Material::Water.is_translucent() && Material::Water.is_fluid());
        assert!(Material::Glass.is_translucent() && !Material::Glass.is_fluid());
        assert!(!Material::Rock.is_translucent());
    }

    #[test]
    fn defaults_are_non_solid() {
        // 走一次 `Default` 与常量两条路径，确保「默认值即空气」对二者都成立。
        let default_physics = BlockPhysics::default();
        let air = BlockPhysics::AIR;
        let rock = BlockPhysics::ROCK;
        assert!(!default_physics.solid);
        assert!(!air.solid);
        assert!(rock.solid, "岩石必须是实体");
        assert!(rock.density_kg_m3 > air.density_kg_m3);
        assert!(!RenderClass::default().emits_geometry());
        assert!(RenderClass::Opaque.emits_geometry());
    }
}
