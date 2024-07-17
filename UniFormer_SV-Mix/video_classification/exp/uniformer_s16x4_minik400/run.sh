work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
 --init_method tcp://localhost:9996 \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR /data/yanbin/kinetics400_mmlab/trainValTest_mini_vid/ \
  DATA.PATH_PREFIX /data/yanbin/ \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.BATCH_SIZE 56 \
  NUM_GPUS 7 \
  UNIFORMER.DROP_DEPTH_RATE 0.1 \
  SOLVER.MAX_EPOCH 110 \
  SOLVER.BASE_LR 1e-4 \
  SOLVER.WARMUP_EPOCHS 10.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 3 \
  TEST.NUM_SPATIAL_CROPS 1 \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path
