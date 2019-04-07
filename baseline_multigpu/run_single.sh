export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export CUDA_VISIBLE_DEVICES='0'

TRAIN_DIR="/root/result/baseline_multigpu_single"
PIPELINE_CONFIG_PATH="/root/configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_3.config"

mpirun -np 1 \
-H localhost:1 \
-bind-to none -map-by slot \
-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
-mca pml ob1 -mca btl ^openib \
python object_detection/custom_main.py \
--train_dir=$TRAIN_DIR \
--pipeline_config_path=$PIPELINE_CONFIG_PATH \


python object_detection/legacy/eval.py \
--logtostderr \
--checkpoint_dir=$TRAIN_DIR \
--eval_dir=$TRAIN_DIR'/eval' \
--pipeline_config_path=$PIPELINE_CONFIG_PATH \
--run_once=True \

