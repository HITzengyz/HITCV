1.训练加入红外模态的map,与可见光相比是否有提升？

改动：添加了val部分的 模态控制参数
改动位置： coco.py-def make_coco_transforms

cd /tmp && python -u /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr/main.py \
  --device npu --dataset_file coco \
  --coco_path /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/data/WaterScenes_real_tir \
  --output_dir /tmp/ws_rgb_only \
  --epochs 10 --batch_size 2 --num_workers 0 \
  --tir_dropout 1 --radar_dropout 1 --tir_strict 1 \
  2>&1 | tee /tmp/ws_rgb_only.log
  
cd /tmp && python -u /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr/main.py \
  --device npu --dataset_file coco \
  --coco_path /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/data/WaterScenes_real_tir \
  --output_dir /tmp/ws_rgb_tir \
  --epochs 10 --batch_size 2 --num_workers 0 \
  --tir_dropout 0 --radar_dropout 1 --tir_strict 1 \
  2>&1 | tee /tmp/ws_rgb_tir.log

  

结果：（缩小到5epoch）

2.基于自定的概率训练模态缺失功能（之前提到过的）tir_valid,radar_valid现在都是默认为1，后续通过一个概率把部分设置为0实现模态缺失

改动：整体修改了dropout内的逻辑，通过指令参数控制模态缺失
--tir_dropout 0.3 --radar_dropout 0.5 30%红外缺失，50%雷达缺失
改动位置：transformer.py-class ModalityDropout；class RadarModalityDropout


3.能否解决加入radar模态后的精度问题



4.优化精度和性能







