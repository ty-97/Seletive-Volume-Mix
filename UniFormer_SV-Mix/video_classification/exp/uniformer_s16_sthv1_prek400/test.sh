work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,6 python tools/run_net.py \
  --cfg $work_path/test.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/sthv1 \
  DATA.PATH_PREFIX /data/tanyi/some_some_v1/img/\
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.BATCH_SIZE 48 \
  TEST.BATCH_SIZE 48 \
  NUM_GPUS 6 \
  UNIFORMER.DROP_DEPTH_RATE 0.2 \
  SOLVER.MAX_EPOCH 60 \
  SOLVER.BASE_LR 2e-4 \
  SOLVER.WARMUP_EPOCHS 5.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TRAIN.ENABLE False \
  OUTPUT_DIR $work_path \
  TEST.CHECKPOINT_FILE_PATH /scratch/tanyi/UniFormer-main_sth240320/video_classification/exp/uniformer_s16_sthv1_prek400/checkpoints/checkpoint_epoch_00050.pyth
