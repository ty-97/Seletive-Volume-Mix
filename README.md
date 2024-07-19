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

| Method | Sth-Sth V1 |  | Sth-Sth V2 |  | Mini-Kinetics |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Acc 1.(%) | $\Delta \operatorname{Acc} 1 .(\%)$ | Acc 1.(%) | $\Delta$ Acc 1.(%) | Acc 1.(%) | $\Delta$ Acc 1.(%) |
| TSM <br> TSM+SV-Mix | 45.5 <br> 47.2 | +1.7 | 59.3 <br> 60.3 | +1.0 | 75.9 <br> 76.6 | +0.7 |
| $\mathrm{R}(2+1) \mathrm{D}$ <br> $\mathrm{R}(2+1) \mathrm{D}+\mathrm{SV}-$ Mix | 45.9 <br> 46.7 | +0.8 | 58.9 <br> 60.3 | +1.4 | 75.5 <br> 76.1 | +0.6 |
| MViTv2 <br> MViTv2+SV-Mix | 57.0 <br> 57.9 | +0.9 | 67.4 <br> 68.6 | +1.2 | 79.3 <br> 79.5 | +0.2 |
| Uniformer <br> Uniformer+SV-Mix | 56.7 <br> $\mathbf{5 7 . 2}$ | +0.5 | 67.7 <br> 68.2 | +0.5 | 79.1 | + |

### Large scale datasets

| Model             | Frame * view   | Top-1 Acc. | Top-5 Acc. | Checkpoint |
| ----------------- | ----------- | ---------- | ----------- | ---------------- |
| H2CN   | 8 * 1  | 53.6%      | 81.4%     |  |
| H2CN   | 16 * 1  | 55.0%      | 82.4%     |  |
| H2CN   | (8+16) * 1  | 56.7%      | 83.2%     |  |

#### Something-Something-V1

| Model             | Frame * view   | Top-1 Acc. | Top-5 Acc. | Checkpoint |
| ----------------- | ----------- | ---------- | ----------- | ---------------- |
| H2CN   | 8 * 1  | 53.6%      | 81.4%     |  |
| H2CN   | 16 * 1  | 55.0%      | 82.4%     |  |
| H2CN   | (8+16) * 1  | 56.7%      | 83.2%     |  |


#### Something-Something-V2
| Model             | Frame * view   | Top-1 Acc. | Top-5 Acc. | Checkpoint |
| ----------------- | ----------- | ---------- | ----------- | ---------------- |
| H2CN   | 8 * 1  | 65.2%      | 89.7%     | |
| H2CN   | 16 * 1  | 66.4%      | 90.1%     | |
| H2CN   | (8+16) * 1  | 67.9%      | 91.2%     |  |


### Kinetics-400

| Model             | Frame * view   | Top-1 Acc. | Top-5 Acc. | Checkpoint |
| ----------------- | ----------- | ---------- | ----------- | ---------------- |
| H2CN   | 8 * 30  | 76.9%      | 93.0%     | [link](https://pan.baidu.com/s/16nf9kad6RClaW3NbInsPBA)(password:defq) |
| H2CN   | 16 * 30  | 77.9%      | 93.3%     | [link](https://pan.baidu.com/s/1SsBSWqJYpM4bnh1ACNqtzg)(password:vert) |
| H2CN   | (8+16) * 30  | 78.7%      | 93.6%     |  |



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
