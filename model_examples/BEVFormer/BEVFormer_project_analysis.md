# BEVFormer（DrivingSDK 模型示例）项目结构化分析

本文基于仓库目录 [/model_examples/BEVFormer](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer) 下现有代码，对 BEVFormer 示例工程的目录结构、训练/评测/推理流程、配置体系、以及 DrivingSDK（mx_driving）/昇腾 NPU 适配点进行梳理，便于后续对 BEVFormer 模型做针对性改造。

## 1. 工程整体定位与边界

- **本目录是 DrivingSDK 的模型示例之一**：示例目标是让 BEVFormer 在昇腾 NPU 上可训练/可评测，并通过 DrivingSDK 提供的高性能算子替换关键算子以获得性能收益（例如 Deformable Attention）。
- **代码组织分两层**：
  - 外层：[/model_examples/BEVFormer](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer) 负责环境准备说明、依赖与打补丁说明、以及 NPU 训练脚本（test/）。
  - 内层：[/model_examples/BEVFormer/BEVFormer](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer) 是“被适配过的” BEVFormer 工程（包含 projects/ 与 tools/），并内置了训练产物目录 work_dirs/、示例数据目录 data/ 等。

## 2. 目录结构速览（建议从这里开始读）

### 2.1 外层目录（DrivingSDK 模型示例包装层）

位于：[/model_examples/BEVFormer](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer)

- [README.md](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/README.md)：本示例的“昇腾侧”使用说明（环境、依赖、训练脚本、性能/精度结果）。
- [DrivingSDK.md](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/DrivingSDK.md)：DrivingSDK 的总体介绍与安装方式（whl 编译/安装、版本配套等）。
- [requirements.txt](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/requirements.txt)：示例 Python 依赖（注意这里只覆盖一部分，mmcv/mmdet/mmdet3d 依赖通常由各自仓库管理）。
- `*_config.patch`：对 mmcv / mmdetection / mmdetection3d / BEVFormer 上游代码的适配补丁（核心是 NPU/算子替换/并行封装）。
- `mmcv/`、`mmdetection/`、`mmdetection3d/`：带补丁的依赖仓（在该示例中以源码形式存在，方便适配 NPU）。
- [test/](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/test)：训练/性能脚本与 NPU 环境变量脚本。
  - [env_npu.sh](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/test/env_npu.sh)：加载 CANN 环境并设置一组训练相关环境变量。
  - [train_full_8p_base_fp32.sh](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/test/train_full_8p_base_fp32.sh) 等：封装 dist_train.sh 的一键训练脚本，并会临时修改 config（epoch、batch size、log interval 等）。

### 2.2 内层目录（BEVFormer 工程主体）

位于：[/model_examples/BEVFormer/BEVFormer](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer)

- [projects/](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects)：OpenMMLab 风格的“项目插件”目录（configs + mmdet3d_plugin）。
  - [projects/configs](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/configs)：配置入口（模型结构、数据集、训练策略）。
    - [projects/configs/bevformer/bevformer_base.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/configs/bevformer/bevformer_base.py)：本示例主要训练配置（R101-DCN + BEVFormer-base）。
  - [projects/mmdet3d_plugin](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin)：插件实现（dataset / model / hook / runner / op 替换等）。
- [tools/](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/tools)：训练/测试入口脚本（对上游工具进行了 NPU 适配）。
  - [tools/train.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/tools/train.py)：训练入口（加载 config、构建模型/数据集、调用自定义训练 API）。
  - [tools/test.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/tools/test.py)：评测入口（分布式评测，使用 NPUDistributedDataParallel）。
- `work_dirs/`、`val/`：训练日志与评测产物（属于运行产物，不是代码逻辑的一部分）。

## 3. 从“跑起来”到“改得动”：训练/评测主链路

### 3.1 训练入口链路

建议先按调用链阅读：

1. 训练脚本（外层封装）  
   - 例如：[train_full_8p_base_fp32.sh](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/test/train_full_8p_base_fp32.sh#L58-L86) 会进入内层目录并调用：
     - `bash ./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py ${world_size}`
2. Python 训练入口  
   - [tools/train.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/tools/train.py#L104-L262)：
     - `cfg = Config.fromfile(args.config)` 解析配置。
     - 若 `cfg.plugin=True`，会通过 `plugin_dir='projects/mmdet3d_plugin/'` 动态导入插件包，以完成自定义 Registry 注册。
     - 构建模型 `build_model(cfg.model, ...)` 与数据集 `build_dataset(cfg.data.train)`。
     - 最终调用 [custom_train_model](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/apis/train.py#L11-L35)。
3. 自定义训练实现（替换并行封装/数据加载等 NPU 细节）  
   - [custom_train_detector](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py#L28-L199)：
     - 通过 [build_dataloader](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/datasets/builder.py#L22-L98) 创建 DataLoader（NPU pin_memory_device）。
     - 使用 `NPUDataParallel / NPUDistributedDataParallel` 包装模型（mmcv.device.npu）。
     - `optimizer = build_optimizer(model, cfg.optimizer)` 构建优化器（本示例 fp32 配置中为 `NpuFusedAdamW`）。
     - `runner.run(...)` 执行训练 workflow。

### 3.2 评测/推理入口链路

- [tools/test.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/tools/test.py#L113-L267)：
  - 与 train 类似：解析 config、导入 plugin、构建 dataset/dataloader、build_model、load_checkpoint。
  - 只支持分布式路径（非 distributed 分支被 `assert False` 阻断）。
  - 使用 `NPUDistributedDataParallel` 包装模型后，调用 `custom_multi_gpu_test` 收集结果并 `dataset.evaluate(...)` 输出指标。

## 4. 配置体系：哪里改模型/数据/训练策略

关键入口配置是 [projects/configs/bevformer/bevformer_base.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/configs/bevformer/bevformer_base.py)：

- **plugin 机制**
  - `plugin = True`
  - `plugin_dir = 'projects/mmdet3d_plugin/'`
  - 作用：让 `projects/mmdet3d_plugin` 下自定义的数据集、模型组件、hook、runner 能注册进 mmcv/mmdet 的注册器。
- **模型结构（model 字段）**
  - `type='BEVFormer'`：对应实现类 [BEVFormer](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py#L20-L292)。
  - `pts_bbox_head` 下 `transformer`：将 encoder/decoder、TemporalSelfAttention、SpatialCrossAttention、CustomMSDeformableAttention 组装起来。
- **数据集与时序队列**
  - `dataset_type = 'CustomNuScenesDataset'`：实现见 [CustomNuScenesDataset](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py#L18-L84)。
  - `queue_length = 4`：每个训练样本会组装 4 帧（含当前帧 + 历史帧）并在 dataset 内打包成一个样本。
- **NPU 优化器与梯度裁剪**
  - fp32 配置里使用 `optimizer.type='NpuFusedAdamW'`（见 [bevformer_base.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/configs/bevformer/bevformer_base.py#L232-L242)）。
  - 对应 mmcv 的 optimizer hook 会识别 fused optimizer 并走 fused clip（见 [optimizer.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/mmcv/mmcv/runner/hooks/optimizer.py#L51-L67)）。

## 5. 模型侧关键实现：时序 BEV 与注意力算子

### 5.1 Detector：BEVFormer 的“时序缓存”与训练时序

实现： [BEVFormer detector](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py)

- **训练时的历史 BEV 构建**
  - dataset 输出的 `img` 是包含多帧的张量：`len_queue = img.size(1)`（见 [forward_train](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py#L179-L234)）。
  - `obtain_history_bev` 里对历史帧 `torch.no_grad()` 逐帧跑 head，得到 `prev_bev`（见 [obtain_history_bev](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py#L158-L178)）。
- **推理时的跨帧状态（video_test_mode）**
  - `self.prev_frame_info` 保存 `prev_bev / scene_token / prev_pos / prev_angle`（见 [__init__](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py#L27-L65)）。
  - `forward_test` 中根据 scene 切换与 can_bus 更新，决定是否使用历史 `prev_bev`（见 [forward_test](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py#L236-L269)）。

### 5.2 Dataset：CustomNuScenesDataset 的“时序拼包”策略

实现： [CustomNuScenesDataset](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py#L18-L84)

- **queue_length 帧采样**
  - `index_list = list(range(index-self.queue_length, index))`，shuffle 后取其中一部分并 append 当前 index（见 [prepare_train_data](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py#L31-L55)）。
  - 这里不是严格的“连续历史帧”采样，而是对窗口内 index 做了随机扰动；如果你要做更稳定的时序训练/复现，优先改这一段。
- **can_bus 的相对位姿编码**
  - `get_data_info` 将 ego2global 的 translation/rotation 写入 `can_bus`，并计算 yaw（见 [get_data_info](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py#L156-L166)）。
  - `union2one` 将同一 scene 内的位姿转换成相对增量（减去上一帧的 pos/angle），并设置 `prev_bev_exists`（见 [union2one](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py#L58-L84)）。

这两处（时序采样策略 + can_bus 编码）直接决定了“模型如何学习跨帧信息”，也是改造 BEVFormer 时最值得优先关注的点。

### 5.3 DrivingSDK 算子替换：Deformable Attention 走 mx_driving

本示例最关键的加速点，是将多尺度可变形注意力的核心计算替换为 DrivingSDK 提供的高性能实现：

- Decoder 侧： [decoder.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/decoder.py#L326-L333)
  - CUDA/NPU 可用时：`mx_driving.multi_scale_deformable_attn(...)`
  - 否则 fallback：`multi_scale_deformable_attn_pytorch(...)`
- TemporalSelfAttention： [temporal_self_attention.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py#L241-L248)
- SpatialCrossAttention： [spatial_cross_attention.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py#L384-L394)

从“改造”角度，这意味着：

- 你后续若要替换/新增注意力结构（例如修改 sampling 规则、改 num_points、融合策略），必须评估：
  - mx_driving 提供的算子签名是否仍可复用；
  - 如不能复用，是否需要回退到 pytorch 实现（会显著影响性能）或在 DrivingSDK 中新增算子实现。

## 6. NPU 适配点清单（在改造时容易踩坑）

### 6.1 并行封装与设备迁移

- 训练使用 [NPUDataParallel / NPUDistributedDataParallel](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py#L70-L91)（mmcv.device.npu）。
- 入口脚本显式 import `torch_npu`（例如 [tools/train.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/tools/train.py#L30-L33)、[tools/test.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/tools/test.py#L26-L29)）。

### 6.2 DataLoader pin_memory_device

- DataLoader 构建时设置 `kwargs = {"pin_memory_device":"npu"}`，并 `pin_memory=True`（见 [builder.py](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/datasets/builder.py#L86-L96)）。
- 如果你后续引入自定义 batch collate、或者改变数据返回结构，要关注 DataContainer / pin_memory 行为是否仍正确。

### 6.3 batch size 相关的硬编码/隐式假设

配置里 `bs_ = 1`，并且在 head 初始化时创建了 `self.bev_mask = torch.zeros((bs, bev_h, bev_w)).npu()`（来自补丁变更；最终代码在当前分支里体现在配置参数 `bs=bs_` 传入 head）。

- 这类“把 batch size 写死在 module 初始化里”的写法，容易在你提升 batch size 时出问题。
- 当前外层脚本会用 `sed` 改 config 的 `samples_per_gpu`，但不一定同步改 `bs_`；因此若要稳定支持 `batch_size>1`，建议统一梳理 `bs_` 的使用点并改为 runtime 推导。

## 7. 对“改造 BEVFormer”最有用的切入点建议

从工程结构看，改造优先级通常是：

1. **改模型结构/算子**：集中在 [projects/mmdet3d_plugin/bevformer](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/bevformer) 下的 detectors / dense_heads / modules。
2. **改时序策略/输入字段**：集中在 [CustomNuScenesDataset](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py) 的 `prepare_train_data/union2one/get_data_info`。
3. **改训练策略/超参**：集中在 configs（尤其是 optimizer、lr schedule、queue_length、bev_h/w、num_query 等）。
4. **改 NPU 性能/并行**：集中在自定义 train/test api、DataLoader、DrivingSDK 算子替换、以及依赖库补丁。

## 8. 附：本示例的“参考实现来源”与版本定位

- 外层 README 标注上游参考实现：fundamentalvision/BEVFormer，commit `66b65f3a...`（见 [README.md](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/README.md#L7-L12)）。
- 内层 [BEVFormer/README.md](file:///root/autodl-tmp/DrivingSDK/model_examples/BEVFormer/BEVFormer/README.md) 是上游项目原始说明，更多偏算法介绍。

