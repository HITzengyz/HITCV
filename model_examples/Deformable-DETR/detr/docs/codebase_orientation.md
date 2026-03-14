# Deformable-DETR 代码库导读（WaterScenes/NPU 变体）

这份文档面向尚不了解 Deformable-DETR 的读者，目标是快速、准确地建立当前仓库的结构认知，并能继续深入到每个核心/支撑 Python 文件。

范围：`model_examples/Deformable-DETR/detr`

## 1）先看运行路径

```text
main.py
  ├─ build_dataset(...)                # datasets/coco.py
  │    └─ make_coco_transforms(...)    # datasets/transforms.py
  │
  ├─ build_model(...)                  # models/deformable_detr.py::build
  │    ├─ build_backbone(...)          # models/backbone.py
  │    ├─ build_deforamble_transformer # models/deformable_transformer.py
  │    └─ SetCriterion + PostProcess   # models/deformable_detr.py
  │
  └─ train_one_epoch / evaluate        # engine.py
       ├─ criterion(...)               # 匈牙利匹配 + 损失计算
       └─ postprocessors["bbox"](...)  # 框解码 + 按图像尺寸缩放
```

## 2）核心代码 vs 支撑代码 vs 非核心产物

| 区域 | 分类 | 为什么重要 |
| --- | --- | --- |
| `models/deformable_detr.py` | 模型核心机制 | 主模型前向、损失与后处理定义都在这里 |
| `models/deformable_transformer.py` | 模型核心机制 | 编码器/解码器与参考点迭代更新逻辑 |
| `models/ms_deform_attn.py` | 模型核心机制 | 编解码器中使用的多尺度可变形注意力 |
| `models/backbone.py` | 模型核心机制 | 特征提取与输入通道适配 |
| `models/matcher.py` | 模型核心机制 | 训练目标的匈牙利分配 |
| `engine.py` | 运行时支撑 | 组织训练/评估循环 |
| `main.py` | 运行时支撑 | 入口、参数、优化器/调度器组装 |
| `datasets/coco.py` | 运行时支撑 | 数据加载，WaterScenes 下 TIR/Radar 构造 |
| `datasets/transforms.py` | 运行时支撑 | 多模态增强、融合与归一化 |
| `datasets/coco_eval.py` | 运行时支撑 | COCO 评估适配层 |
| `util/misc.py`, `util/box_ops.py` | 运行时支撑 | 张量打包、分布式辅助、框运算 |
| `tools/*.py` | 支撑/诊断 | 调试与可视化脚本，不在核心前向路径上 |
| `models/ops/*` | 历史/可选路径 | 上游 CUDA 扩展路径；当前代码主要在 `models/ms_deform_attn.py` 中使用 `mx_driving` |
| `visualization/`, `artifacts/`, `result/` | 非核心生成产物 | 运行输出图片，不是实现真值来源 |
| `test/output*`, `kernel_meta/` | 非核心生成产物 | 运行日志与调试转储 |
| `checkpoints/` | 非核心生成产物 | 训练产出的权重，不是实现代码 |

## 3）这个仓库变体的特有点

该目录不是纯上游拷贝，包含项目定制行为：

- `main.py` 中有 NPU 相关运行选项和融合优化器使用。
- 数据与变换中有 WaterScenes 的模态融合约定：
  - RGB + TIR + `tir_valid`
  - 可选 Radar 通道 + `radar_valid`
- `backbone.py` 通过替换 `conv1` 输入通道支持非 3 通道输入。
- 仓库内包含较多实验输出文件。

## 4）新手阅读顺序

### 快速入门（15-30 分钟）

1. `main.py`（看全局流程和参数）
2. `models/deformable_detr.py`（看 `build`、`DeformableDETR`、`SetCriterion`）
3. `engine.py`（看每个 iteration/epoch 的实际流程）

### 进阶（30-90 分钟）

1. `models/deformable_transformer.py`
2. `models/ms_deform_attn.py`
3. `datasets/coco.py`
4. `datasets/transforms.py`

### 深入（90 分钟以上）

1. `matcher.py`、`util/box_ops.py`、`datasets/coco_eval.py`
2. `tools/` 下用于对齐与评估排障的脚本
3. 对照 `Deformable-DETR_npu.patch` 理解相对上游改动

## 5）先改哪里 vs 先别改哪里

### 优先可改（当你要改行为逻辑时）

- `datasets/coco.py`
- `datasets/transforms.py`
- `engine.py`
- `models/deformable_detr.py`
- `models/deformable_transformer.py`
- `models/backbone.py`

### 建议先别改（通常不是实现目标）

- `visualization/`、`artifacts/`、`result/`
- `test/output*`、`kernel_meta/`、`checkpoints/`
- 顶层 `fusion_result.json`
- notebook checkpoint 或临时笔记文件

## 6）“与模型核心无关”的仓库内示例

下面这些文件对调试或留档有用，但不属于模型核心算法实现：

- `detr/visualization/000*_overlay.png`
- `detr/artifacts/waterscenes-radar-overlay-viz/*`
- `detr/test/output*/train.log`
- `detr/kernel_meta/kernel_meta_temp_*/**/*`
- `model_examples/Deformable-DETR/fusion_result.json`

## 7）覆盖基线（Section-2 核心/支撑 Python 文件）

本次深度讲解只覆盖 Section-2 中“核心模型机制 + 运行时支撑”的 Python 文件，基线如下（12 个）：

1. `models/deformable_detr.py`
2. `models/deformable_transformer.py`
3. `models/ms_deform_attn.py`
4. `models/backbone.py`
5. `models/matcher.py`
6. `engine.py`
7. `main.py`
8. `datasets/coco.py`
9. `datasets/transforms.py`
10. `datasets/coco_eval.py`
11. `util/misc.py`
12. `util/box_ops.py`

## 8）逐文件详细讲解（核心 + 支撑）

### 8.0 统一阅读模板

每个文件按同一结构讲解：

- 文件角色（在整体流水线里的位置）
- 关键符号（核心类/函数）
- 调用关系（谁调用它、它调用谁）
- 数据契约（输入/输出）
- 形状注记（仅写可静态推导者）

---

### 8.1 `models/deformable_detr.py`（核心）

**文件角色**
- 这是“检测头 + 损失 + 后处理 + build 装配”总入口。

**关键符号**
- `class DeformableDETR`
- `class SetCriterion`
- `class PostProcess`
- `def build(args)`

**调用关系**
- 被 `models/__init__.py::build_model` 间接调用。
- 在训练时被 `engine.py` 调用前向和 loss。
- `build(args)` 内部调用 `build_backbone`, `build_deforamble_transformer`, `build_matcher`。

**数据契约**
- 前向输入：`NestedTensor`（图像张量 + padding mask）。
- 前向输出：`pred_logits`, `pred_boxes`，可选 `aux_outputs`、`enc_outputs`。
- 训练损失：由 `SetCriterion` 计算分类与框回归相关 loss。

**形状注记（可推导）**
- 输入图像：`samples.tensors` 为 `[B, C_in, H, W]`。
- 文件中有显式断言：`C_in == expected_in_channels`。
- 最终预测：
  - `pred_logits`: `[B, Nq, C_cls]`
  - `pred_boxes`: `[B, Nq, 4]`
- 其中：
  - `B`=batch size
  - `Nq`=`num_queries`（默认 300，可配置）
  - `C_cls` 取决于 `num_classes` 与当前分类头定义（由 `build(args)` 决定）

---

### 8.2 `models/deformable_transformer.py`（核心）

**文件角色**
- 定义 Deformable Transformer 的 encoder/decoder 主体。

**关键符号**
- `class DeformableTransformer`
- `class DeformableTransformerEncoderLayer`
- `class DeformableTransformerEncoder`
- `class DeformableTransformerDecoderLayer`
- `class DeformableTransformerDecoder`
- `def build_deforamble_transformer(args)`

**调用关系**
- 在 `deformable_detr.py` 的 `build(args)` 中创建实例。
- 在 `DeformableDETR.forward` 中被调用。
- 内部依赖 `MSDeformAttn`（来自 `models/ms_deform_attn.py`）。

**数据契约**
- 输入是多尺度特征列表（`srcs`）及对应 mask/position embedding。
- 输出 decoder hidden states、reference points，以及 two-stage 分支相关结果。

**形状注记（可推导）**
- 单层输入特征：`src_l` 为 `[B, D, H_l, W_l]`。
- flatten 后：`src_flatten` 为 `[B, S, D]`，`S = Σ_l(H_l * W_l)`。
- `spatial_shapes`: `[L, 2]`，`L` 为特征层数。
- 非 two-stage 时 query embedding split 后扩展为 `[B, Nq, D]`。
- decoder intermediate 输出（`return_intermediate_dec=True`）：`hs` 的逻辑形态为 `[N_dec, B, Nq, D]`。

---

### 8.3 `models/ms_deform_attn.py`（核心）

**文件角色**
- 实现多尺度可变形注意力模块本体（本仓使用 `mx_driving` 算子路径）。

**关键符号**
- `class MSDeformAttn`
- `def _is_power_of_2(n)`

**调用关系**
- 被 `deformable_transformer.py` 的 encoder/decoder layer 使用。
- 内部调用 `mx_driving.multi_scale_deformable_attn(...)` 完成核心算子。

**数据契约**
- 输入：`query`, `reference_points`, `input_flatten`, `input_spatial_shapes`, `input_level_start_index`。
- 输出：与 query 同 token 维度的更新特征。

**形状注记（可推导）**
- `query`: `[B, Len_q, D]`
- `input_flatten`: `[B, Len_in, D]`
- `value` reshape 后：`[B, Len_in, H_head, D_head]`
- `sampling_offsets`: `[B, Len_q, H_head, L, P, 2]`
- `attention_weights`: `[B, Len_q, H_head, L, P]`
- 输出：`[B, Len_q, D]`

---

### 8.4 `models/backbone.py`（核心）

**文件角色**
- 封装 ResNet backbone、mask 对齐、position encoding 拼接。

**关键符号**
- `class FrozenBatchNorm2d`
- `class BackboneBase`
- `class Backbone`
- `class Joiner`
- `def build_backbone(args)`

**调用关系**
- 在 `deformable_detr.py::build` 中被创建。
- 在 `DeformableDETR.forward` 中被调用获取多尺度特征。

**数据契约**
- 输入：`NestedTensor`。
- 输出：`out`（多尺度 `NestedTensor` 列表）+ `pos`（每层位置编码）。

**形状注记（可推导）**
- 输入：`[B, C_in, H, W]` + mask `[B, H, W]`。
- 典型中间层输出通道（当前硬编码 for resnet50 family）：`[512, 1024, 2048]`。
- 对应 stride（默认）：`[8, 16, 32]`（若 dilation 配置，最后一层 stride 会变化）。

---

### 8.5 `models/matcher.py`（核心）

**文件角色**
- 负责预测与 GT 的匈牙利匹配（LSAP），为 loss 计算提供配对索引。

**关键符号**
- `class HungarianMatcher`
- `def build_matcher(args)`

**调用关系**
- 在 `deformable_detr.py::build` 中创建。
- 在 `SetCriterion.forward` 中调用。

**数据契约**
- 输入：`outputs`（`pred_logits`, `pred_boxes`）和 `targets`。
- 输出：每个 batch 元素的 `(prediction_indices, target_indices)`。

**形状注记（可推导）**
- `pred_logits`: `[B, Nq, C_cls]`。
- `pred_boxes`: `[B, Nq, 4]`。
- flatten 后参与代价矩阵计算：`[B*Nq, ...]`。
- 代价矩阵按 batch reshape 后求解。

---

### 8.6 `engine.py`（支撑）

**文件角色**
- 训练与评估循环的执行器。

**关键符号**
- `def train_one_epoch(...)`
- `def evaluate(...)`

**调用关系**
- 由 `main.py` 的 epoch 循环调用。
- 训练时调用 `model(samples)` 和 `criterion(outputs, targets, ...)`。
- 评估时调用 `postprocessors['bbox']` 与 `CocoEvaluator`。

**数据契约**
- 输入：模型、criterion、dataloader、optimizer 等。
- 输出：训练/评估统计字典。

**形状注记（可推导）**
- 循环中 `samples` 是 `NestedTensor`；`samples.tensors` 形状 `[B, C_in, H, W]`。
- 评估时 `orig_target_sizes` 由 target 堆叠得到，形状 `[B, 2]`（`[h, w]`）。
- 其余 shape 随模型输出定义传递。

---

### 8.7 `main.py`（支撑）

**文件角色**
- 训练入口：参数定义、数据集构建、模型构建、优化器/调度器、训练/评估主循环。

**关键符号**
- `def get_args_parser()`
- `def _preset_waterscenes_fusion_args(args)`
- `def main(args)`

**调用关系**
- 脚本入口 `if __name__ == '__main__': main(args)`。
- 调用 `build_dataset`, `build_model`, `train_one_epoch`, `evaluate`。

**数据契约**
- 通过 args 约束输入通道与模态配置（含 TIR/Radar 相关参数）。
- 管理 checkpoint 的读写与恢复。

**形状注记（可推导）**
- 通过 `_preset_waterscenes_fusion_args` 和 dataset build，`in_channels` 可为：
  - `5`（RGB+TIR+tir_valid）
  - `5 + radar_channels + 1`（含 radar 与 radar_valid）
- 具体 `H/W` 取决于 transforms 与数据。

---

### 8.8 `datasets/coco.py`（支撑）

**文件角色**
- 数据读取主入口，处理 COCO 与 WaterScenes 两种布局，并负责 TIR/Radar 数据装配。

**关键符号**
- `class CocoDetection`
- `class ConvertCocoPolysToMask`
- `def make_coco_transforms(...)`
- `def build(image_set, args)`

**调用关系**
- 被 `datasets/__init__.py::build_dataset` 调用。
- `__getitem__` 内调用 transform pipeline（`datasets/transforms.py`）。

**数据契约**
- 返回 `(img, target)`；`img` 经过 transform 后是 fused tensor。
- `target` 包含 boxes/labels 等检测标注以及模态字段（`tir`, `tir_valid`, 可选 `radar_k`, `radar_valid`）。

**形状注记（可推导）**
- Radar map 默认构造为 `[radar_channels, H, W]`（默认 `radar_channels=4`）。
- 经 `Normalize` 融合后图像通道数：
  - 非 WaterScenes 模式：`[5, H, W]`
  - WaterScenes 模式：`[5 + radar_channels + 1, H, W]`

---

### 8.9 `datasets/transforms.py`（支撑）

**文件角色**
- 统一实现 RGB/TIR/Radar 的空间增强、模态 dropout、融合与归一化。

**关键符号**
- 函数：`crop`, `hflip`, `resize`, `pad`
- 类：`ToTensor`, `ModalityDropout`, `RadarModalityDropout`, `Normalize`, `Compose` 等

**调用关系**
- 由 `datasets/coco.py::make_coco_transforms` 组装并在 dataset `__getitem__` 中调用。

**数据契约**
- 输入：PIL RGB 图 + `target`（含可选模态）
- 输出：融合后 tensor 图像 + 同步更新后的 target

**形状注记（可推导）**
- `Normalize.__call__` 在融合后执行 `torch.cat(parts, dim=0)`，得到 `[C_fused, H, W]`。
- valid 标记（`tir_valid` / `radar_valid`）会被扩展为 `[1, H, W]` 的二值 map。

---

### 8.10 `datasets/coco_eval.py`（支撑）

**文件角色**
- 将模型输出转换为 COCO API 可评估格式，并支持分布式聚合。

**关键符号**
- `class CocoEvaluator`
- `def convert_to_xywh(boxes)`
- `def merge(...)`, `def create_common_coco_eval(...)`

**调用关系**
- 在 `engine.evaluate` 中构建并更新。
- 内部调用 pycocotools 的 `COCOeval`。

**数据契约**
- 输入：按 `image_id` 组织的预测结果字典。
- 输出：COCO 评估统计（AP/AR 等）。

**形状注记（可推导）**
- `convert_to_xywh` 输入 boxes 为 `[N, 4]`，输出同为 `[N, 4]`（语义从 `xyxy` 转 `xywh`）。
- 评估矩阵内部维度由 COCOeval 决定；静态阅读不建议硬编码具体轴长度。

---

### 8.11 `util/misc.py`（支撑）

**文件角色**
- 提供训练日志、分布式辅助、`NestedTensor` 容器、batch collate 等通用基础设施。

**关键符号**
- `class MetricLogger`, `class SmoothedValue`
- `class NestedTensor`
- `def nested_tensor_from_tensor_list(...)`
- `def collate_fn(...)`
- 分布式相关函数（`init_distributed_mode`, `reduce_dict`, `all_gather` 等）

**调用关系**
- 被 `main.py`, `engine.py`, `datasets/transforms.py`, `models/*` 广泛引用。

**数据契约**
- `nested_tensor_from_tensor_list` 将不同尺寸 `[C, H_i, W_i]` pad 成 batch tensor + mask。

**形状注记（可推导）**
- 输出 batch tensor：`[B, C, H_max, W_max]`
- mask：`[B, H_max, W_max]`（padding 区域为 True）
- 其余分布式函数对 shape 无新增约束。

---

### 8.12 `util/box_ops.py`（支撑）

**文件角色**
- 检测框格式转换、IoU/GIoU 计算、mask-to-box。

**关键符号**
- `box_cxcywh_to_xyxy`, `box_xyxy_to_cxcywh`
- `box_iou`, `generalized_box_iou`
- `masks_to_boxes`

**调用关系**
- 被 `matcher.py`, `deformable_detr.py`, `transforms.py` 等调用。

**数据契约**
- 大多数函数输入/输出的 box 最后一维固定为 4。

**形状注记（可推导）**
- `box_iou(boxes1, boxes2)`：
  - `boxes1`: `[N, 4]`
  - `boxes2`: `[M, 4]`
  - 输出 IoU: `[N, M]`
- `generalized_box_iou` 输出同样是 `[N, M]`。

## 9）关键路径的形状传播（跨文件总览）

下面给出仅基于静态代码可推导的主链路形状：

```text
Dataset/Transforms
  image -> fused tensor [B?, C_in, H, W]  (sample 级别通常先是 [C_in, H, W])

Collate (util/misc.py)
  list[[C_in, H_i, W_i]] -> NestedTensor.tensors [B, C_in, H_max, W_max]
                          -> NestedTensor.mask    [B, H_max, W_max]

Backbone (models/backbone.py)
  [B, C_in, H, W] -> 多尺度特征: [B, C_l, H_l, W_l], l=1..L

Input projection + Transformer
  -> 统一通道后 [B, D, H_l, W_l]
  -> flatten 后 [B, S, D], S = Σ(H_l*W_l)
  -> decoder states hs [N_dec, B, Nq, D]

Detection head (models/deformable_detr.py)
  -> pred_logits [B, Nq, C_cls]
  -> pred_boxes  [B, Nq, 4]

Eval postprocess (engine.py + PostProcess)
  -> 按 target_sizes [B,2] 缩放为像素坐标
  -> 每图 top-k 结果（k=100，来自当前 PostProcess 实现）
```
