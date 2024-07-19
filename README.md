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




## Runing 
Our implementation includes 3 project: _SV-Mix_, _SlowFast_SV-Mix_Mvit_ and _UniFormer_SV-Mix_. For runing CNN models and Transfomer models (only on small scale datasets), we use _SV-Mix_ by command:
```
cd SV-Mix
python3 train_net.py --num-gpus 4 --config-file experiments/ssv1_tsm.yaml
```
Config files for other datasets and models can be found in experiments folder.

For runing MvitV2 on large scale datasets, we use _SlowFast_SV-Mix_Mvit_ project. The running command is as:

```
cd SlowFast_SV-Mix_Mvit
python tools/run_net.py --cfg configs/SSv1/MVITv2_S_16x4.yaml --init_method tcp://localhost:9997
```
Config files for other datasets and models can be found in configs folder.

For runing Uniformer on large scale datasets, we use _UniFormer_SV-Mix_ project. The running command is as:

```
cd UniFormer_SV-Mix/video_classification
bash exp/uniformer_s16_sthv1_prek400/run.sh
```
Config files for other datasets and models can be found in exp folder.





## Citing
```bash
@article{tan2023selective,
  title={Selective Volume Mixup for Video Action Recognition},
  author={Tan, Yi and Qiu, Zhaofan and Hao, Yanbin and Yao, Ting and He, Xiangnan and Mei, Tao},
  journal={arXiv preprint arXiv:2309.09534},
  year={2023}
}
```

