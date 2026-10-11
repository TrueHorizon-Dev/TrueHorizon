//! 光标射线拾取：体素级 DDA 步进。

use block_schema::BlockId;
use glam::DVec3;
use planet_coords::VoxelCoord;
use world_contracts::WorldView;

/// 一条世界射线。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// 起点（世界坐标，米）。
    pub origin_m: DVec3,
    /// 方向（无需归一化，内部会处理）。
    pub direction: DVec3,
}

impl Ray {
    /// 构造。
    pub fn new(origin_m: DVec3, direction: DVec3) -> Self {
        Self {
            origin_m,
            direction,
        }
    }

    /// 起点沿方向前进 `distance_m` 后的点。
    pub fn at(&self, distance_m: f64) -> DVec3 {
        self.origin_m + self.direction.normalize_or_zero() * distance_m
    }
}

/// 一次命中。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// 被命中的体素。
    pub voxel: VoxelCoord,
    /// 该体素的方块。
    pub block: BlockId,
    /// 起点到命中面的距离（米）。
    pub distance_m: f64,
    /// 命中面的外法线（体素格子坐标下的 -1/0/1）。
    pub normal: [i32; 3],
    /// 命中点世界坐标（米）。
    pub point_m: DVec3,
}

/// 体素 DDA（Amanatides & Woo）。
///
/// 复杂度 `O(途经体素数)`，与射线长度无关；每次只跨越一个体素边界，
/// 因此**不会漏掉薄壁**（这正是逐体素采样做不到的）。
///
/// 起始体素若已是实体则立即命中（`distance_m == 0`）。
pub fn raycast<W: WorldView + ?Sized>(world: &W, ray: Ray, max_distance_m: f64) -> Option<RayHit> {
    if !max_distance_m.is_finite() || max_distance_m <= 0.0 {
        return None;
    }
    let direction = ray.direction.normalize_or_zero();
    if direction == DVec3::ZERO || !direction.is_finite() {
        return None;
    }
    let edge = world.voxel_edge_m();
    if !edge.is_finite() || edge <= 0.0 {
        return None;
    }

    let mut voxel = VoxelCoord::from_world_m(ray.origin_m);
    let mut normal = [0i32; 3];

    // 起始体素检查。
    let start_block = world.block_at(voxel);
    if !start_block.is_air() {
        return Some(RayHit {
            voxel,
            block: start_block,
            distance_m: 0.0,
            normal,
            point_m: ray.origin_m,
        });
    }

    let axis_value = |voxel: VoxelCoord, axis: usize| -> i64 {
        match axis {
            0 => voxel.x,
            1 => voxel.y,
            _ => voxel.z,
        }
    };
    let origin_axis = |origin: DVec3, axis: usize| -> f64 {
        match axis {
            0 => origin.x,
            1 => origin.y,
            _ => origin.z,
        }
    };

    let mut step = [0i64; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];

    for axis in 0..3 {
        let component = match axis {
            0 => direction.x,
            1 => direction.y,
            _ => direction.z,
        };
        if component == 0.0 {
            continue;
        }
        step[axis] = if component > 0.0 { 1 } else { -1 };
        t_delta[axis] = (edge / component).abs();
        let current = axis_value(voxel, axis) as f64 * edge;
        let boundary = if component > 0.0 {
            current + edge
        } else {
            current
        };
        t_max[axis] = ((boundary - origin_axis(ray.origin_m, axis)) / component).max(0.0);
    }

    let mut distance;
    loop {
        let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            0
        } else if t_max[1] <= t_max[2] {
            1
        } else {
            2
        };
        if !t_max[axis].is_finite() {
            return None;
        }
        distance = t_max[axis];
        if distance > max_distance_m {
            return None;
        }

        match axis {
            0 => {
                voxel.x += step[0];
                normal = [(-step[0]) as i32, 0, 0];
            }
            1 => {
                voxel.y += step[1];
                normal = [0, (-step[1]) as i32, 0];
            }
            _ => {
                voxel.z += step[2];
                normal = [0, 0, (-step[2]) as i32];
            }
        }
        t_max[axis] += t_delta[axis];

        let block = world.block_at(voxel);
        if !block.is_air() {
            return Some(RayHit {
                voxel,
                block,
                distance_m: distance,
                normal,
                point_m: ray.at(distance),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// 测试用「点云世界」：只记录被显式放置的体素。
    struct SparseWorld {
        voxels: HashMap<(i64, i64, i64), BlockId>,
    }

    impl SparseWorld {
        fn with(blocks: &[(i64, i64, i64)]) -> Self {
            let mut voxels = HashMap::new();
            for &(x, y, z) in blocks {
                voxels.insert((x, y, z), BlockId::new(1));
            }
            Self { voxels }
        }
    }

    impl WorldView for SparseWorld {
        fn block_at(&self, voxel: VoxelCoord) -> BlockId {
            self.voxels
                .get(&(voxel.x, voxel.y, voxel.z))
                .copied()
                .unwrap_or(BlockId::AIR)
        }

        fn has_chunk(&self, _coord: world_contracts::ChunkCoord) -> bool {
            true
        }

        fn surface_height_m(&self, _dir: DVec3) -> Option<f64> {
            None
        }
    }

    #[test]
    fn hits_the_nearest_voxel_along_the_ray() {
        let world = SparseWorld::with(&[(0, 0, 0), (0, 0, 5)]);
        let ray = Ray::new(DVec3::new(0.5, 0.5, -4.0), DVec3::Z);
        let hit = raycast(&world, ray, 100.0).expect("应当命中");
        assert_eq!(hit.voxel, VoxelCoord::new(0, 0, 0));
        assert_eq!(hit.distance_m, 4.0, "从 z=-4 到 z=0 边界");
        assert_eq!(hit.normal, [0, 0, -1], "从 -Z 侧进入");
        assert!((hit.point_m.z - 0.0).abs() < 1e-9);
    }

    #[test]
    fn thin_walls_are_never_skipped() {
        // 单层薄壁：逐体素采样很容易漏掉，DDA 不会。
        let world = SparseWorld::with(&[(3, 0, 0)]);
        for offset in [-0.9, -0.4, 0.0, 0.4, 0.9] {
            let ray = Ray::new(DVec3::new(-10.0, 0.5 + 0.0 * offset, 0.5), DVec3::X);
            let hit = raycast(&world, ray, 100.0).expect("薄壁必须被命中");
            assert_eq!(hit.voxel.x, 3);
        }
    }

    #[test]
    fn respects_max_distance_and_misses() {
        let world = SparseWorld::with(&[(0, 0, 0)]);
        // 朝远处打，超出最大距离。
        assert!(raycast(&world, Ray::new(DVec3::new(5.5, 5.5, 0.5), DVec3::X), 2.0).is_none());
        // 起点即为实体，距离为 0。
        let inside = raycast(&world, Ray::new(DVec3::new(0.5, 0.5, 0.5), DVec3::X), 10.0).unwrap();
        assert_eq!(inside.distance_m, 0.0);
        // 非法输入必须安全返回 None，而不是死循环。
        assert!(raycast(&world, Ray::new(DVec3::ZERO, DVec3::ZERO), 10.0).is_none());
        assert!(raycast(&world, Ray::new(DVec3::ZERO, DVec3::X), 0.0).is_none());
    }

    #[test]
    fn negative_coordinates_are_handled() {
        let world = SparseWorld::with(&[(-3, -3, -3)]);
        let ray = Ray::new(DVec3::new(-0.5, -2.5, -2.5), -DVec3::X);
        let hit = raycast(&world, ray, 100.0).expect("负坐标方向也必须命中");
        assert_eq!(hit.voxel, VoxelCoord::new(-3, -3, -3));
        assert_eq!(hit.normal, [1, 0, 0], "沿 -X 前进，命中面的法线朝 +X");
    }
}
