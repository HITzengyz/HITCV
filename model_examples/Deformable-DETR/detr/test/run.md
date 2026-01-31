  服务器上运行示例：

  cd /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr                                                                                               
                                                                                                                                                                   
 python3 test/infer_visualize_rgb_tir.py --img_name 00098.jpg --data_root /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/data/waterscenes-coco        
  --ckpt /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr/test/output/checkpoint.pth --score_thr 0.3 --out_dir /root/autodl-tmp/DrivingSDK/         
  model_examples/Deformable-DETR/detr/result   