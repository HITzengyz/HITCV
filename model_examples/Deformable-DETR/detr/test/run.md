  服务器上运行示例：

  cd /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr
  训练：
  cd /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr
    bash test/train_8p_full.sh --data_path=/root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/data/waterscenes-coco --epochs=1 --batch_size=10 --nproc_per_node=1                                                                                                   
推理：
cd /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr

 python3 test/infer_visualize_rgb_tir.py --img_name 00098.jpg --data_root /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/data/waterscenes-coco        
  --ckpt /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr/test/output/checkpoint.pth --score_thr 0.3 --out_dir /root/autodl-tmp/DrivingSDK/         
  model_examples/Deformable-DETR/detr/result   


  1. 新建一个 screen 会话                                                                                                                                                                                                                                                                            
  screen -S detr_train                                                                                                                                  
                                                                                                                                                        
  2. 在 screen 里运行训练                                                                                                                               
                                                                                                                                                        
  cd /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr                                                                                    
  bash test/train_8p_full.sh --data_path=/root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/data/waterscenes-coco --epochs=10 --batch_size=10    
  --nproc_per_node=1                                                                                                                                    
                                                                                                                                                        
  3. 退出但保持后台运行                                                                                                                                 
     按：Ctrl + A 然后按 D                                                                                                                              
  4. 查看后台会话                                                                                                                                       
                                                                                                                                                        
  screen -ls                                                                                                                                            
                                                                                                                                                        
  5. 重新进入会话                                                                                                                                       
                                                                                                                                                        
  screen -r detr_train                                                                                                                                  
                                                                                                                                                        
  6. 结束会话                                                                                                                                           
     在会话里按 Ctrl + C 停止训练，然后输入 exit 或 Ctrl + D。                                                                                          
                                                                                                                                                        
  如果希望训练日志同时写文件，建议用：                                                                                                                                                                                                                                                 
  bash test/train_8p_full.sh ... 2>&1 | tee /root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr/test/output/train_screen.log       