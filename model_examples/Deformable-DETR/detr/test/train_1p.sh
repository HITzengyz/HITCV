#!/bin/bash

node_size=1
batch_size=2
epochs=50
master_port=29500
data_path="/root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/data"

for para in "$@"
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
    if [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    fi
    if [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
    if [[ $para == --master_port* ]];then
        master_port=`echo ${para#*=}`
    fi
done

if [[ "$data_path" == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi

if [ ! -d "${data_path}" ]; then
    echo "[Error] data_path not found: ${data_path}"
    exit 1
fi

echo "[INFO] Start setting ENV VAR"

if command -v msnpureport >/dev/null 2>&1; then
    msnpureport -g error -d 0
    msnpureport -g error
    msnpureport -e disable
fi

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export HCCL_WHITELIST_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# Avoid PyTorch warnings caused by invalid thread env vars (e.g., OMP_NUM_THREADS=0).
if [[ -z "${OMP_NUM_THREADS}" || "${OMP_NUM_THREADS}" -le 0 ]]; then
    export OMP_NUM_THREADS=8
fi
if [[ -z "${MKL_NUM_THREADS}" || "${MKL_NUM_THREADS}" -le 0 ]]; then
    export MKL_NUM_THREADS=8
fi

echo "[INFO] Finish setting ENV VAR"

cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}

if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
output_path_dir=${test_path_dir}/output
if [ -d ${output_path_dir} ]; then
  rm -rf ${output_path_dir}
fi
mkdir -p ${output_path_dir}

start_time=$(date +%s)
echo "start_time=$(date -d @${start_time} "+%Y-%m-%d %H:%M:%S")"

log_file=${output_path_dir}/train_1p.log

torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=${master_port} main.py \
    --output_dir=${output_path_dir} \
    --batch_size=${batch_size} \
    --epochs=${epochs} \
    --coco_path=${data_path} 2>&1 | tee ${log_file}

end_time=$(date +%s)
echo "end_time=$(date -d @${end_time} "+%Y-%m-%d %H:%M:%S")"
e2e_time=$(( $end_time - $start_time ))

avg_time=`grep "Epoch: .* Total time"  ${log_file} | tail -n 5 | grep -oP "[0-9]+\.[0-9]+" | awk '{sum+=$1; count++} END {if(count>0) print sum/count}'`
mAP=`grep "Average Precision.* IoU=0\.50\:0.95.* all" ${log_file} |awk -F "=" '{print $NF}'|awk 'END {print}'`
avg_fps=`awk 'BEGIN{printf "%.3f\n", '$batch_size'*'${node_size}'/'$avg_time'}'`

echo "[INFO] Final Result"
echo " - End to End Time : ${e2e_time}s"
echo " - Final Performance images/sec :  ${avg_fps}"
echo " - Final mAP(IoU=0.50:0.95) : ${mAP}"
