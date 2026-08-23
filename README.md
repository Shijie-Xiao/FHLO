# FHLO — Fast Hurricane Large-ensemble ODE

基于 FAST（Frictional, Axisymmetric, Steady-state Tropical cyclone）物理约束
ODE 的飓风强度集合预报框架。核心流程：IBTrACS best track → 1h 插值 →
ERA5/OISST 环境场提取（含严格 vortex surgery 去涡旋）→ FAST 参考模式积分。

## 目录结构

```
FHLO/
├── run.py / run.slurm / run.README.md   # 端到端 pipeline（详见 run.README.md）
├── config.txt                            # pipeline 配置（风暴、数据源、并行度）
│
├── prep/                                 # 数据准备
│   ├── IBtracs_datasets.py               #   IBTrACS 下载 → 6h best track CSV
│   ├── prepare_complete_training_data.py #   1h 插值 + 环境场提取 → dataset.pkl
│   └── prepare_ensemble_storm.py         #   集合成员轨迹拼接（best track + 合成）
│
├── physics/                              # FAST 模式
│   ├── run_fast_reference.py             #   单轨迹参考 ODE（48h init + F 衰减预报）
│   ├── run_fast_physics_torch.py         #   GPU/批量版本
│   ├── Fast.py / env.py / track.py       #   模式核心与强迫
│   └── constants.py / utils.py
│
├── common/                               # 共享库
│   ├── thermo_table/                     #   熵表热力学（chi 饱和熵亏计算）
│   ├── vortex_inversion/                 #   vortex_lib：涡旋剔除（pyamg+pyshtools）
│   └── util/                             #   球面几何、海盆、常数
│
├── tracks/                               # 集合轨迹统计模型（Markov 采样）
│   ├── config.py                         #   风暴自动发现（TIGGE ECMWF）
│   ├── read_files.py / build_pairs.py    #   TIGGE 读取 → 6h 速度对
│   ├── train_markov.py / sample_tracks.py#   条件 Markov 训练与采样
│   └── gefs_tracks.py                    #   GEFS 轨迹源
│
├── ensemble/                             # 大集合预报
│   ├── gefs_ens_adapter.py               #   GEFS GRIB2 → ERA5 等价场适配器
│   ├── run_prep_gefs_ens1000.py          #   1000 成员集合 prep
│   ├── run_prep_div1000.py / predict_chi_s_div1000.py
│   └── run_ode_from_chis.py              #   从 ML 预测 chi/s 跑 ODE 集合
│
├── download/                             # 数据获取与裁剪
│   ├── crop_beryl_sample.py              #   ERA5/OISST 按日期裁剪（全空间）
│   ├── crop_era5.slurm                   #   裁剪 sbatch 包装
│   └── download*.py                      #   HURDAT2/ATCF/TIGGE 下载
│
├── precalc_data/                         # 静态场（Cd、陆地、混合层、层结、水深）
├── data/                                 # 本地数据（gitignore，含 demo 裁剪）
└── environment.yml                       # conda 环境 fast_ml
```

## 环境安装

```bash
conda env create -f environment.yml
conda activate fast_ml
```

关键依赖：`pyamg` + `pyshtools`（vortex surgery）、`xarray`/`netcdf4`（数据）、
`scipy`（插值/样条）、`tcpyPI`（潜在强度，可选）。

## 快速开始（以 Beryl 2024 为例）

```bash
# 1) best track（已有 data/ibtracs 时可跳过）
python prep/IBtracs_datasets.py

# 2) 裁剪 demo 环境场数据（按日期、全空间，约 90 s）
sbatch download/crop_era5.slurm NA 2024 2024181N09320_BERYL

# 3) 一键 pipeline：prep（vortex surgery）+ FAST ODE
sbatch run.slurm --storms 2024181N09320_BERYL
```

结果落盘在 `data/ibtracs/NA/2024/2024181N09320_BERYL/`：
`{STORM}_dataset.pkl`（环境场+轨迹数据集）、`fast_reference.csv`、
`fast_reference.png`（强度对比图）。

Beryl 参考：MAE 25.0 kts，相关系数 0.890，峰值 130 vs 145 kts。

## 数据源与配置

`config.txt` 支持按风暴切换数据源：

| 用途 | 可选源 | 本地目录 |
|---|---|---|
| SST | ERA5 SSTK / OISST (GHRSST) | `data/era5` / `data/oisst` |
| 环境场 (T/Q/U/V/Z/MSL) | ERA5 / GEFS | `data/era5` |
| 集合采样轨迹 | ECMWF TIGGE / GEFS | `data/tracks/tigge` |

本地数据均为**按日期裁剪、全空间保留**（NA 源档案 lat 0–80°N / lon 0–360°），
保证 vortex surgery 不会碰到数据边界；demo 数据自包含，可直接上传分发。

## 关键物理约定

- **chi（饱和熵亏）**：thermo_table 熵表反演，annulus 200–800 km 中层
  T/q 环平均，失效时取物理上限 4.0
- **环境风/风切变**：严格 `vortex_lib.vortex_surgery`（Helmholtz 分解剔除
  涡旋分量后取风暴中心处风），非环形平均
- **FAST ODE**：48 h 初始化（V 跟踪 Vtarget 反演的 V_axisym，累积 F 强迫，
  window=12 平滑），预报段 F 按 exp[-(lead/24h)²] 衰减，4 子步/h
- **V_axisym ↔ V_max**：axi_to_max_wind 二分反演（含平移速度与切变投影）

## 复现与扩展

- 换风暴：`config.txt` 改 `storms=`，或 `python run.py --storms <NAME>`
- 换 SST 源：`python run.py --sst OISST`
- 多风暴并行：`config.txt` 设 `n_workers=`（风暴级进程池）
- 集合预报：`ensemble/run_prep_gefs_ens1000.py`（GEFS 1000 成员）
  或 `tracks/train_markov.py` + `sample_tracks.py`（Markov 合成轨迹）
