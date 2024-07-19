# Selective Volume Mixup for Video Action Recognition
This is an official implementaion of paper "Selective Volume Mixup for Video Action Recognition". [`Paper link`](https://arxiv.org/pdf/2309.09534)
<div align="center">
  <img src="model.png" width="700px"/>
</div>


## Updates
### Oct 11, 2022
* Release this V1 version (the version used in paper) to public. Complete codes and models will be released soon.

## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Code](#code)
- [Performance](#performance)
- [Train](#Train)
- [Test](#Test)
- [Contibutors](#Contributors)
- [Citing](#Citing)
- [Acknowledgement](#Acknowledgement)

## Prerequisites

The conda environment can be built by ```conda env create -f environment.yaml```

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).

## Data Preparation

 Please download the datasets and origenize the anotation files as the following form:
 ```
...
[path to video file or frame folder] [num of frames] [category index]
...
```


## Code


Our implements are partitially based on [SlowFast](https://github.com/facebookresearch/SlowFast/), [Uniformer](https://github.com/Sense-X/UniFormer) codebases


## Performance



### Large scale datasets

| Method           | Sth-Sth V1     | Sth-Sth V2     | Mini-Kinetics  |
|------------------|----------------|----------------|----------------|
| TSM              | 45.5           | 59.3           | 75.9           |
| TSM+SV-Mix       | **47.2(+1.7)** | **60.3(+1.0)** | **76.6(+0.7)** |
| R(2+1)D          | 45.9           | 58.9           | 75.5           |
| R(2+1)D+SV-Mix   | **46.7(+0.8)** | **60.3(+1.4)** | **76.1(+0.6)** |
| MViTv2           | 57.0           | 67.4           | 79.3           |
| MViTv2+SV-Mix    | **57.9(+0.9)** | **68.6(+1.2)** | **79.5(+0.2)** |
| Uniformer        | 56.7           | 67.7           | 79.1           |
| Uniformer+SV-Mix | **57.2(+0.5)** | **68.2(+0.5)** |                |



### Small scale datasets

| Method           | UCF101         | Diving48       | EGTEA GAZE+    |
|------------------|----------------|----------------|----------------|
| TSM              | 85.2           | 77.6           | 63.5           |
| TSM+SV-Mix       | **88.4(+3.2)** | **80.2(+2.6)** | **65.5(+2.0)** |
| ViViT            | 87.3           | 70.0           | 57.3           |
| ViViT+SV-Mix     | **88.3(+1.0)** | **76.2(+6.2)** | **62.1(+4.8)** |
| MViTv2           | 90.0           | 80.7           | 66.5           |
| MViTv2+SV-Mix    | **92.2(+2.2)** | **83.8(+3.1)** | **67.8(+1.3)** |
| VideoSwin        | 93.6           | 78.7           | 67.0           |
| VideoSwin+SV-Mix | **96.6(+3.0)** | **82.8(+4.1)** | **68.8(+1.8)** |
| Uniformer        | 93.2           | 83.1           | 69.7           |
| Uniformer+SV-Mix | **97.1(+3.9)** | **85.0(+1.9)** | **71.9(+2.2)** |


### Diving48
| Model             | Frame * view   | Top-1 Acc.  | Checkpoint |
| ----------------- | ----------- | ----------  | ---------------- |
| H2CN   | 16 * 1  | 87.0%      |  |



### EGTEA Gaze
| Model             | Frame * view * clip    | Split1 |  Split2 | Split3 |
| ----------------- | ----------- | ---------- | ----------- | ----------- |
| H2CN  | 8 * 1 * 1  | 66.2%     | 63.9%    | 60.5%  |


## Train 

```
python train.py
```


## Test 
We use the test code and protocal of repo [TDN](https://github.com/MCG-NJU/TDN)

- For center crop single clip, the processing of testing can be summarized into 2 steps:
    1. Run the following testing scripts:
        ```
        CUDA_VISIBLE_DEVICES=0 python3 test_models_center_crop.py something \
        --archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8  \
        --test_crops=1 --batch_size=16  --gpus 0 --output_dir <your_pkl_path> -j 4 --clip_index=0
        ```
    2. Run the following scripts to get result from the raw score:
        ```
        python3 pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir <your_pkl_path>  
        ```
- For 3 crops, 10 clips, the processing of testing can be summarized into 2 steps: 
    1. Run the following testing scripts for 10 times(clip_index from 0 to 9):
        ``` 
        CUDA_VISIBLE_DEVICES=0 python3 test_models_three_crops.py  kinetics \
        --archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8 \
        --test_crops=3 --batch_size=16 --full_res --gpus 0 --output_dir <your_pkl_path>  \
        -j 4 --clip_index <your_clip_index>
        ```
    2. Run the following scripts to ensemble the raw score of the 30 views:
        ```
        python pkl_to_results.py --num_clips 10 --test_crops 3 --output_dir <your_pkl_path> 
        ```


## Citing
```bash
@article{H2CN2022,
  title={Hierarchical Hourglass Convolutional Network for Efficient Video Classification},
  author={Yi Tan, Yanbin Hao, Hao Zhang, Shuo Wang, Xiangnan He},
  journal={MM 2022},
}
```

## Acknowledgement
Thanks for the following Github projects:
- https://github.com/yjxiong/temporal-segment-networks
- https://github.com/mit-han-lab/temporal-shift-module
- https://github.com/MCG-NJU/TDN
