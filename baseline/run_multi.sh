export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export CUDA_VISIBLE_DEVICES='0,1,2,3'

TRAIN_DIR="/root/result/baseline_multi"
PIPELINE_CONFIG_PATH="/root/configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2.config"

python object_detection/legacy/train.py \
--logtostderr \
--train_dir=$TRAIN_DIR \
--pipeline_config_path=$PIPELINE_CONFIG_PATH \
--num_clones=4 \


python object_detection/legacy/eval.py \
--logtostderr \
--checkpoint_dir=$TRAIN_DIR \
--eval_dir=$TRAIN_DIR'/eval' \
--pipeline_config_path=$PIPELINE_CONFIG_PATH \
--run_once=True \
