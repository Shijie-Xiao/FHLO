# run.py — FHLO 端到端 Pipeline

`run.py` 将三个阶段串成一条命令，以飓风为单位并行处理多个风暴：

```
best-track (IBTrACS) → 1h 插值 → 环境场提取（vortex surgery）→ FAST ODE
```

| 阶段 | 实现文件 | 输入 | 输出（每风暴目录内） |
|---|---|---|---|
| 1. ibtracs | `prep/IBtracs_datasets.py` | NOAA NCEI 在线 CSV | `track_intensity_6h.csv` |
| 2. prep | `prep/prepare_complete_training_data.py` | 6h track + 本地 ERA5/OISST | `track_intensity_1h.csv`、`{STORM}_dataset.pkl` |
| 3. ode | `physics/run_fast_reference.py` | dataset.pkl | `fast_reference.csv`、`fast_reference.png` |

## 快速开始

```bash
# 环境要求见 environment.yml（核心依赖 pyamg / pyshtools 用于 vortex surgery）
conda activate fast_ml

# 默认按 config.txt 跑（风暴列表、数据源都在里面）
python run.py

# 指定风暴 / 覆盖数据源 / 只跑部分阶段
python run.py --storms 2024181N09320_BERYL,2024279N21265_MILTON
python run.py --sst OISST                 # SST 从 ERA5 换成 OISST
python run.py --stage prep,ode            # 跳过 IBTrACS 下载
python run.py --overwrite                 # 重算已有 pkl
python run.py --list                      # 列出将处理的风暴
```

## Slurm 后台运行

```bash
sbatch run.slurm                          # 按 config.txt
sbatch run.slurm --storms 2024181N09320_BERYL --sst OISST
```

任务输出在 `logs/run_<jobid>.out`。QOS 默认 premium（立即调度），
可在 `run.slurm` 顶部改。

## config.txt 字段

| 字段 | 取值 | 说明 |
|---|---|---|
| `basins` | `NA` / `EP` / `ALL` | 风暴海盆 |
| `year_start` / `year_end` | 年份 | best-track 年份范围 |
| `storms` | 风暴目录名，逗号分隔；`ALL` = 全部 | 要跑的风暴 |
| `sst_source` | `ERA5` / `OISST` | SST 数据源二选一 |
| `env_source` | `ERA5` / `GEFS` | 环境场数据源 |
| `track_source` | `ECMWF` / `GEFS` | 采样轨迹来源（集合预报） |
| `era5_dir` / `oisst_dir` | 路径 | 本地裁剪数据目录 |
| `n_workers` | 整数 | 风暴级并行进程数 |

数据源字段留空或缺失时用默认值（ERA5/ERA5/ECMWF）。

## 本地数据准备（demo 数据裁剪）

Pipeline 只读本地数据，不访问 CFS 档案。首次使用需先裁剪：

```bash
# ERA5 + OISST 按日期裁剪（全空间保留），Beryl 18 天约 90 秒
python download/crop_beryl_sample.py NA 2024 2024181N09320_BERYL
sbatch download/crop_era5.slurm NA 2024 2024181N09320_BERYL   # 或后台跑
```

输出：
- `data/era5/{T,Q,U,V,Z}_{YYYYMMDD}.nc` — 日文件，多层，**全空间**
  （NA 源档案范围 lat 0–80°N / lon 0–360°）
- `data/era5/{SSTK,MSL,BLH}_{YYYYMM}.nc` — 月文件
- `data/oisst/{YYYYMMDD}.nc` — 逐日 GHRSST SST（°C，原生网格 0–60°N/-100–0°）

为什么只按日期裁剪：vortex surgery 需要风暴中心周围完整 25° 环域，
区域裁剪会让靠近边界的风暴（如 Beryl 起步 8.9°N）直接失败。

## 并行模型

- **风暴间并行**：`ProcessPoolExecutor(n_workers)`，每个风暴一个进程
- **风暴内串行**：单风暴沿时间轴推进，vortex surgery 的中间缓存
  （Helmholtz 分解结果）在同一风暴的时间步间自然复用
- 每 1h 步约 1.5–2 s（vortex surgery 主导），Beryl（313 步）约 8 分钟单进程

## 输出说明

`{STORM}_dataset.pkl` 键：

| 键 | 形状 | 内容 |
|---|---|---|
| `spatial_3d` | (1, T, 5, 7, 72, 72) | T/Q/U/V/Z @ 7 层，72×72 空间窗 |
| `spatial_2d` | (1, T, 2, 72, 72) | SSTK、MSL |
| `scalars` | (1, T, 4) | alpha, beta, gamma, vp（潜在强度） |
| `chi_ref` / `s_ref` / `xs_ref` | (1, T, 1) | 饱和熵亏、风切变、二者乘积 |
| `env_wnds` | (1, T, 4) | u/v @ 250/850 hPa（vortex surgery 后） |
| `v_gt` / `lats` / `lons` / `times` | (T,) | 观测强度、位置、时间 |
| `cd_ref` / `blh_ref` / `utran` / `vtran` | (1, T, 1) | 拖曳系数、边界层高度、平移速度 |
| `sst_source` | str | 本次使用的 SST 数据源 |

`fast_reference.csv`：逐步 `vp_kts, v_obz_kts, v_fast_kts, v_max_kts, m`；
`fast_reference.png`：三线对比图（FAST / 观测 / 潜在强度）。

## Beryl 2024 参考结果

- prep：313 步（6h→1h），vortex surgery 0% 边界失败
- ODE：MAE 25.0 kts，相关系数 0.890，峰值 130 vs 145 kts
- 单风暴总耗时 ≈ 9 分钟（premium 节点单进程）
