#CUDA_VISIBLE_DEVICES=7 
python tools/run_net.py \
  --cfg configs/SSv2/MVITv2_S_16x4.yaml --init_method tcp://localhost:9997