# EgoSmith LeRobot 数据集格式说明（无图像版）

面向使用方的格式文档。本文描述 EgoSmith 发布的 **无图像（labels-only）LeRobot 数据集**：
只含手部位姿、相机位姿、语言标注和"每一帧对应原始数据集哪一帧"的索引，不含任何像素和深度。
图像需要使用方按源数据集自己的协议获取原始媒体后，用 §6 的索引自行回填。

版本：LeRobot **v3.0**（`meta/info.json` 的 `codebase_version`），已用官方 `lerobot==0.6.1` 加载器验收。
每个源数据集（EgoDex、EPIC-KITCHENS、Ego4D……）是一个**独立**的 LeRobot 数据集，不混装，因为各自许可不同。
转换入口为 `scripts/build/wds_to_lerobot.py`，字段说明见本文后续章节。

---

## 1. 目录结构

```
<dataset_root>/
├── meta/
│   ├── info.json                      # 总帧数、episode 数、fps、feature 定义、路径模板、splits；video_path 为 null
│   ├── stats.json                     # 每个 feature 的全局统计（min/max/mean/std/q01…q99），归一化用
│   ├── tasks.parquet                  # 任务词表：task 字符串 ↔ task_index
│   ├── episodes/chunk-XXX/file-XXX.parquet   # 每个 episode 一行的元数据（§5）
│   ├── source_frames.parquet          # 回填索引（§6）；只要转换时传了 descriptor_manifest 就会写
│   ├── release.json                   # 发布等级、许可信息（§8）；仅 batch 的 release 步骤写出
│   └── egosmith_provenance.json       # 转换参数：坐标系、布局、源 shard 列表、映射来源与覆盖率
├── data/chunk-XXX/file-XXX.parquet    # 帧级数据，每行一帧；每文件约 100 MB，一个 episode 一个 row group
├── LICENSES/<源数据集>/LICENSE.txt, NOTICE.txt   # 源数据集许可全文与版权声明；仅 batch 的 release 步骤写出（internal 等级不写）
└── README.md                          # 来源、许可义务、我们做过的修改、加载方式；仅 batch 的 release 步骤写出
```

`release.json`、`LICENSES/` 和根目录 `README.md` 只由 `scripts/build/wds_to_lerobot_batch.py` 的 `release` 步骤写出；
单独运行 `wds_to_lerobot.py` 得到的数据集没有这三项。

没有 `videos/` 目录，`info.json` 的 `features` 里没有图像 feature。`chunk-XXX/file-XXX` 编号由 LeRobot 规定：
`chunks_size = 1000`，每文件约 100 MB，一个 episode 不会跨文件。

## 2. 帧级 features（`data/*.parquet`，每行一帧）

| feature | dtype | shape | 含义 |
|---|---|---|---|
| `observation.state` | float32 | (74,) | 本体状态，EgoSteer 74 维布局，见 §3 |
| `action` | float32 | (74,) | 动作 = **下一帧**的 state，布局与 state 相同 |
| `observation.hand_presence` | int64 | (1,) | 手是否在画面中的位掩码：0 无、1 左手、2 右手、3 双手 |
| `observation.camera.head_world2cam` | float32 | (16,) | 该帧头部相机的世界→相机 4×4 齐次矩阵，**行优先**展平；点按列向量左乘 |
| `observation.mano` | float32 | (110,) | 可选。左手 PCA45+betas10，右手 PCA45+betas10。源 shard 带 `mano.npy` 时才有；当前批次没有 |
| `source_frame_index` | int64 | (1,) | 仅转换时传了 `--source_map` 才有。该帧对应**原始数据集媒体**里的帧号，按 `source_media_fps` 计数（§6）。-1 只按整个 episode 出现：episode 在 source map 中没有可用行时整段为 -1 |
| `source_frame_observed` | bool | (1,) | 仅转换时传了 `--source_map` 才有。该帧的标注是否直接来自被解码过的原始帧。为 false 时标注由相邻已观测帧插值得到（§6） |
| `timestamp` | float32 | (1,) | `frame_index / fps`，episode 内相对时间 |
| `frame_index` | int64 | (1,) | episode 内帧号，从 0 起 |
| `episode_index` | int64 | (1,) | episode 号，全数据集内从 0 起连续 |
| `index` | int64 | (1,) | 全局帧号，全数据集内连续 |
| `task_index` | int64 | (1,) | 指向 `meta/tasks.parquet` 的行 |

没有图像、没有深度。转换器在校验阶段强制检查输出中不存在任何深度 feature、深度列或 `videos/` 下的深度目录；
"无图像"由转换配置保证（labels_only 等级强制 `include_video = false`），校验阶段不单独检查图像 feature。

## 3. `observation.state` / `action` 的 74 维布局

与 EgoSteer 真机数据集完全一致的切片定义；人手数据没有机器人本体，前 26 维用常数填充。

| 切片 | 内容 | 单位 | 人手数据取值 |
|---|---|---|---|
| `[0:7]` | 左臂 7 关节角 | rad | 填充（默认 0） |
| `[7:14]` | 右臂 7 关节角 | rad | 填充（默认 0） |
| `[14:20]` | 左手 6 自由度电机归一化值 | – | 填充（默认 0） |
| `[20:26]` | 右手 6 自由度电机归一化值 | – | 填充（默认 0） |
| `[26:29]` | 左腕位置 xyz | m | MANO 腕关节（joint 0）位置 |
| `[29:35]` | 左腕旋转 rot6d | – | 腕坐标系到参考系的旋转 |
| `[35:38]` | 右腕位置 xyz | m | 同左 |
| `[38:44]` | 右腕旋转 rot6d | – | 同左 |
| `[44:59]` | 左手 5 指尖 xyz（拇/食/中/无名/小 × xyz） | m | |
| `[59:74]` | 右手 5 指尖 xyz | m | |

填充值记录在 `meta/egosmith_provenance.json` 的 `pad_value`。`info.json` 里每一维都有名字（`features["observation.state"]["names"]`）。

**rot6d 定义**：旋转矩阵 R 的前两列按列优先展开，`[R00, R10, R20, R01, R11, R21]`。还原时对两列做 Gram-Schmidt 正交化，第三列取叉积。

**参考坐标系（重要）**：默认 `hand_frame = world`。state **和** action 都表达在该 episode 的 **SLAM 世界系**里，
action 是下一帧的手在同一世界系下的位姿。以观测帧为锚的训练用法（世界系手部 + 逐行 w2c）取同一行的
`observation.camera.head_world2cam`，reshape 成 4×4 得到 `T_w2c`，则第 *t* 帧头部相机系下的点为
`p_cam = T_w2c · [p_world; 1]`，旋转同理左乘 `T_w2c[:3,:3]`（相机坐标系：x 右、y 下、z 前，OpenCV 约定）。

也可显式选 `hand_frame = camera`：第 *t* 帧的 state 和 action 已被变换到**第 t 帧头部相机**系（EgoSteer "世界系 ≡ 头部相机系"），
此时不要再乘一次 w2c；想回到 SLAM 世界系用 `p_world = inv(T_w2c) · [p_cam; 1]`。实际使用的模式记录在
`meta/egosmith_provenance.json` 的 `hand_frame`。

投影到图像：`u = fx·x/z + cx`，`v = fy·y/z + cy`，内参见 §5 的 `calibration/head_intrinsics`，
对应的图像分辨率见 `calibration/head_image_size`。

## 4. tasks 与 instructions

- `meta/tasks.parquet`：pandas 风格，索引列 `task`（字符串），列 `task_index`。每帧的 `task_index` 指向一行。
- **task 是数据集名**（例如 `egodex`、`egocentric_100k`），整个数据集通常只有一个 task（转换器默认 `task_source = dataset_name`，
  batch 驱动默认再把 job 名作为 `task_name`）。下游把 LeRobot 转回 WebDataset 的工具（不在本仓库中）会把 `tasks[0]`
  当作 `dataset_name`，所以这里放数据集名而不是句子。
- 语言在 episode 元数据里：`instructions`（`List[str]`）是该 episode 标注的全部指令（1 到 5 条，由粗到细），
  `language`（str）是源 shard 里单独给出的那一句（没有时为空串）。训练时从 `instructions` 里取语言输入。

## 5. episode 元数据（`meta/episodes/*.parquet`，每个 episode 一行）

| 列 | 类型 | 含义 |
|---|---|---|
| `episode_index`, `length` | int64 | episode 号、帧数 |
| `tasks` | List[str] | 单元素列表，即数据集名（§4） |
| `instructions` | List[str] | 全部标注指令 |
| `language` | str | 源 shard 里单独给出的一句指令；没有时为空串 |
| `split` | str | `train` / `val`；episode 按 split 分组排列，`info.json.splits` 给出连续区间 |
| `source_media` | str | 原始数据集里的媒体标识：相对源数据集根目录的文件路径，或该数据集自己的 take / 序列 id（例如 `factory021/factory_021_worker_005_0169.mp4`）。无法重建时为空串 |
| `source_media_fps` | float64 | 该媒体的原生帧率，`source_frame_index` 按它计数 |
| `source_media_frame_start`, `source_media_frame_end` | int64 | 该 episode 覆盖的原始帧区间 [start, end)，与首末帧的 `source_frame_index` 一致；无可用映射时为 -1（此时 `source_media_fps` 为 0） |
| `source_media_undistort` | str | 生成标注时对原始帧施加过的映射名，回填时要对原始帧做同样的映射：空串表示原始帧直接可用；`opencv_fisheye_knew_k` 表示按 `calibration/source_camera` 做 OpenCV 鱼眼去畸变（新内参取 K）。无可用映射时为空串 |
| `calibration/source_camera` | float64 × 8 | 原始媒体相机参数 `[fx, fy, cx, cy, k1, k2, k3, k4]`；未知或无可用映射时全为 NaN |
| `clip_id` | str | EgoSmith 流程内的 clip 编号（内部追溯用） |
| `source_dataset`, `source_episode_index`, `source_frame_start` | str / int64 | 中间 WebDataset 的名字、episode 号、首帧帧号（内部追溯用） |
| `calibration/head_intrinsics` | float64 × 9 | 头部相机内参 3×3 行优先 `[fx,0,cx, 0,fy,cy, 0,0,1]`，按 `calibration/head_image_size` 的分辨率给出 |
| `calibration/head_image_size` | int64 × 2 | `[width, height]`，内参对应的图像分辨率；回填脚本不做缩放，抽出的帧应与该分辨率一致，否则内参需按比例换算 |
| `calibration/head_world2cam` | float64 × 16 | 仅 `hand_frame = camera` 时存在，恒为单位阵（参考系就是头部相机）；默认 world 模式下不写该列，逐帧 w2c 见 `observation.camera.head_world2cam` |
| `data/chunk_index`, `data/file_index` | int64 | 该 episode 在哪个 data 文件 |
| `dataset_from_index`, `dataset_to_index` | int64 | 该 episode 的全局帧号区间 [from, to) |
| `stats/<feature>/{min,max,mean,std,count,q01,q10,q50,q90,q99}` | list | 该 episode 内每个 feature 的统计 |
| `meta/episodes/chunk_index`, `meta/episodes/file_index` | int64 | 本行所在的元数据文件 |

## 6. 与原始数据集的帧对应关系

这是无图像版的核心：每一帧都能对回原始数据集的某个媒体文件的某一帧。

- **媒体**：episode 的 `source_media` 指向原始数据集里的一个文件（相对源数据集根目录）或 take id，按源数据集发布时的目录结构写，
  使用方从源方拿到数据后不需要改名。
- **帧号**：帧的 `source_frame_index` 是该媒体按 `source_media_fps` 解码时的第几帧（从 0 起）。
  如果拿到的媒体帧率与 `source_media_fps` 不同（例如源方后来重新编码），先换算成时间 `t = source_frame_index / source_media_fps` 再定位。
- **观测与插值**：EgoSmith 流程可能以低于 30 fps 的频率解码原始视频（例如 5 fps），再把手部与相机标签插值到 30 fps。
  `source_frame_observed = true` 的帧，其标注直接来自被解码的原始帧；`false` 的帧，标注是相邻两个观测帧之间插出来的，
  `source_frame_index` 仍给出它在原始媒体里对应的帧号。只想要"真观测"的使用方按这一列过滤即可。
- **一个 episode 对应一段连续区间**：`[source_media_frame_start, source_media_frame_end)`，episode 内 `source_frame_index` 单调不减。
- **无法对应的情形**：-1 不会逐帧出现——有可用映射行的 episode 每一帧都有帧号（按 source map 的起始帧与步长推算）；
  某个 episode 在 source map 中没有可用行时，默认（`source_map_policy = drop`）整个 episode 被丢弃，
  `source_map_policy = keep` 时保留，其 `source_media` 为空串、`source_frame_index` 全为 -1；转换时没有传 `--source_map` 则没有
  `source_frame_index` / `source_frame_observed` 两列，`source_media` 为空串。`meta/egosmith_provenance.json` 的
  `source_mapping` 字段说明映射来源与覆盖率。

回填图像：`lerobot_rehydrate_video.py` **不直接读取原始视频**，它读取的是按 EgoSmith prepare 阶段方式预先抽好的 JPEG 帧。
使用方需要先从原始媒体自行抽帧，放成如下目录结构，再运行脚本：

```
<frames_root>/<clip_id>/<frame_name>      # 默认 --layout clip_id
<frames_root>/<clip_name>/<frame_name>    # --layout clip_name
```

其中 `clip_id` / `clip_name` / `frame_names` 取自本数据集的 `meta/source_frames.parquet`（只要转换时传了
`descriptor_manifest` 就会写这个文件，与是否带图像无关；batch 的 labels_only 等级要求必须有它；缺失时脚本直接报错）。抽帧要与 EgoSmith prepare 阶段一致（相同的解码帧率、裁剪与帧命名），
使 `frame_names` 中每个名字都能在对应目录下找到（名字不带 `.jpg` 时会再尝试 `<stem>.jpg`）。

`source_frames.parquet` 每个 episode 一行，列为：`episode_index`、`clip_id`、`source_id`、`clip_name`、`media_name`、
`storage_kind`、`fps`、`width`、`height`、`frame_count`、`frame_names`（该 clip 抽帧的文件名列表，只保留基名）、
`frame_start_name`（episode 第一帧的文件名）、`frame_index_in_clip`（`list<int64>`，长度等于 episode 帧数：第 t 行对应的
clip 内帧号，即 `frame_names` 的下标）、`descriptor_extra`（JSON 字符串）。episode 的源帧号允许有空洞（转换时只打 warning），
回填脚本按 `frame_index_in_clip` 逐行取 `frame_names[i]`，因此有空洞的 episode 也逐帧对齐。旧版转换器写的文件没有
`frame_index_in_clip` 列，此时脚本退回从 `frame_start_name` 起连续取帧并打印警告（无法发现空洞导致的错位）；
`frame_start_name` 为空或不在 `frame_names` 中时直接报错。然后运行：

```bash
python scripts/build/lerobot_rehydrate_video.py --labels_root <本数据集> --frames_root <frames_root> --output_dir <带图像数据集>
```

脚本按 episode 顺序读取这些 JPEG，**不做缩放**（视频分辨率取自第一张帧），用 ffmpeg 编码为 h264
（默认 `crf 18`、`GOP 15`、`yuv420p`，可用 `--crf` / `--gop` / `--preset` 调整）写入 `videos/observation.images.head/`，
并在 `info.json`、episode 元数据和 `stats.json` 里补上图像 feature。`--output_dir` 必须不存在或为空。
回填后的数据集可以用同一套校验工具复核。

## 7. 用 lerobot 加载

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="local/epic_kitchens_labels", root="/path/to/epic_kitchens_labels")
item = ds[0]
item["observation.state"]         # torch.float32 [74]
item["action"]                    # torch.float32 [74]
item["task"]                      # 数据集名；语言指令在 episode 元数据的 instructions 里
item["source_frame_index"]        # torch.int64 [1]

# 取历史 16 步 state、未来 32 步 action（30 fps）
fps = ds.fps
delta = {
    "observation.state": [-(k) / fps for k in range(15, -1, -1)],
    "action": [k / fps for k in range(32)],
}
ds = LeRobotDataset(repo_id="local/epic_kitchens_labels", root="/path/to/epic_kitchens_labels", delta_timestamps=delta)
item = ds[100]
item["action"].shape              # [32, 74]
item["action_is_pad"]             # [32] bool，越过 episode 末尾的位置为 True
```

无图像版在 lerobot 里就是一个没有 video feature 的普通数据集，不需要任何解码后端。
不用 lerobot 也能读：`data/*.parquet` 用 pyarrow/pandas 直接打开，`observation.state` 等是变长 `list<float>` 列，每行长度固定为 74。

## 8. 发布等级与许可

经 batch `release` 步骤发布的数据集，其 `meta/release.json` 记录：

| 字段 | 含义 |
|---|---|
| `tier` | 本文对应 `labels_only`：只发标注，像素由使用方按源协议自行获取 |
| `license` | 源数据集许可（SPDX 标识或 `custom`） |
| `derived_license` | 本数据集的许可，按源许可推导（ShareAlike 来源 → CC-BY-SA-4.0，非商业来源 → CC-BY-NC-4.0，宽松来源 → CC-BY-4.0） |
| `source_name`, `source_url`, `copyright`, `notes` | 来源、获取原始媒体的入口与义务说明 |

`LICENSES/` 里附源许可全文；`README.md` 说明我们对源数据做过的修改（重采样、切片、手部位姿为 EgoSmith 估计值、
语言标注为生成值、重打包为 LeRobot 布局）以及回填步骤。

## 9. 与 EgoSteer 真机数据集的对照

| | EgoSteer 真机 | EgoSmith 人手无图像版（本文） |
|---|---|---|
| 图像 | 头部 + 胸部，RGB + 深度 | 无；提供头部相机内参、分辨率和原始帧索引，回填后为头部 RGB |
| `state/action` | 74 维，全部有效 | 74 维，[0:26] 填充，[26:74] 有效 |
| 参考系 | 世界系 ≡ 头部相机系（相机固定） | 默认 SLAM 世界系（相机运动，附每帧 w2c）；`hand_frame = camera` 时为每帧头部相机系 |
| `calibration/*` | 头/胸内参、手眼标定、world2cam | 头部内参、分辨率；`hand_frame = camera` 时另有单位阵 world2cam |
| task | 193 个正式任务名（分类） | 数据集名；语言指令在 `instructions` |
| 额外列 | 无 | `observation.hand_presence`、`observation.camera.head_world2cam`、`source_frame_index`、`source_frame_observed`、`source_media*` |

两者能用同一套加载代码读取；训练代码只需按 `robot_type`（`egosteer_realman_bimanual` / `egosmith_human`）区分处理前 26 维。

## 10. 一致性保证

batch 驱动（`scripts/build/wds_to_lerobot_batch.py`）对每个数据集依次运行以下检查，报告在 `<dataset_root>/_verification/`：
结构校验（含上文的无深度检查）；配置了参考数据集时，与其（例如 EgoSteer 真机数据集）做结构逐项对拍；
每一帧的 state/action 从源 WebDataset 的 lowdim 用独立实现重新推导比对（带视频时还会把视频帧解码后与源 JPEG 做 PSNR 比对，
并可输出手部投影 overlay 视频供人工查看）；安装了官方 lerobot 时，用其加载器抽样加载并做跨 episode 边界的窗口采样。
这些检查都以源 WebDataset 为基准，**不会**按 `source_frame_index` 回原始媒体取帧比对；帧索引映射的覆盖率见
`meta/egosmith_provenance.json` 的 `source_mapping`。
