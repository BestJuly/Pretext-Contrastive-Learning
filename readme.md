# Pretext-Contrastive Learning: Toward Good Practices in Self-supervised Video Representation Leaning
Codes are in refactoring.    
Currently support PCL(VCP).

## Highlights
1. This paper represents a joint optimization method in self-supervised video representation learning, which can achieve high performance without proposing new pretext tasks;
2. The effectiveness of our proposal is validated by 3 pretext task baselines and 4 different network backbones;
3. The proposal is flexible enough to be applied to other methods.


## Requirements
> This is my experimental environment when preparing this demo code. 
- Ubuntu 18.04.4 LTS
- conda 4.8.4
- PyTorch 1.4.0
- python  3.8.3
- cuda 10.1
- accimage 


## Usage
### Data preparation
I used resized RGB frames from this [repo](https://github.com/feichtenhofer/twostreamfusion). Frames of videos in UCF101 and HMDB51 datasets can be downloaded directly without decoding.

> Tips: There is a folder called `TSP_Flows` inside `v_LongJump_j18_c03 folder` in UCF101 dataset and you may meet a problem if you do not handle this. One solution is to delete this folder.

The folder architecture is like `path/to/dataset/jpegs_256/video_id/frames.jpg`.

Then, you need to edit `datasets/ucf101.py` and `datasets/hmdb51.py` to specify the path for dataset. Please change `*_dataset_path` in line #19. 

### Training self-supervised learning part with our PCL
```
python train_vcp_contrast.py
```
Default settings are
- Method: PCL (VCP)
- Backbone: R3D
- Modality: Res
- Augmentation: RandomCrop, ColorJitter, RandomGrayScale, GaussianBlur, RandomHorizontalFlip
- Dataset: UCF101
- Split: 1

These settings are also fixed for the following process, we do not need to specify `--model=r3d --modality=res`.

> The training will take around 42 hours on one V100 based on our experimental environment.

Models will be saved to `./logs/exp_name`. Here, `exp_name` is directly generated by its corresponding settings.

### Evaluation using video retrieval
```
python retrieve_clips.py --ckpt=/path/to/ssl/best_model --dataset=ucf101
```

### Fine-tuning on video recognition
```
python ft_classify.py --ckpt=/path/to/ssl/best_model --dataset=ucf101
```
The testing process will automatically run after training is done.

## Results
### Video retrieval
Our PCL outperform a set of methods by a large margin. Here we list results using Resnet-18-3D as network backbone. For more results, please refer to our paper.

Methods  | Backbone | Top1 | Top5 | Top10 | Top20 | Top50
--- | --- | --- | --- | --- | --- | --- | 
*Random* | R3D-18 | 15.3 | 25.1 | 32.1 | 40.8 | 53.7 
3DRotNet | R3D-18 | 14.2 | 25.2 | 33.5 | 43.7 | 59.5 
VCP | R3D-18 | 22.1 | 33.8 | 42.0 | 51.3 | 64.7 
RTT | R3D-18 | 26.1 | 48.5 | 59.1 | 69.6 | 82.8 
PacePred | R3D-18 | 23.8 | 38.1 | 46.4 | 56.6 | 69.8 
IIC | R3D-18 | 36.8 | 54.1 | 63.1 | 72.0 | 83.3 
PCL (3DRotNet) | R3D-18 | 33.7 | 53.5 | 64.1 | 73.4 | 85.0 
PCL (VCP) | R3D-18 | **55.1** | **71.2** | **78.9** | **85.5** | **92.3**


### Video recognition
The table lists recognition results on UCF101 and HMDB51 datasets. Other results are from corresponding paper. Because this is the most widely used metrics, we show results based on 4 different network backbones. 

Methods in this table do not contain those using other data modalities such as sound and text.

Method |  Date | Pre-train | ClipSize | Network | UCF  | HMDB   
--- | --- | --- | --- | --- | --- | --- |
OPN      | 2017   | UCF    | $227^2$               | VGG     | 59.6 | 23.8  
DPC          | 2019   | K400   | $16\times224^2$     | R3D-34  | 75.7 | 35.7  
CBT         | 2019   | K600+  | $16\times112^2$     | S3D     | 79.5 | 44.6  
SpeedNet  | 2020   | K400   | $64\times224^2$     | S3D-G   | 81.1 | 48.8  
MemDPC         | 2020   | K400   | $40\times224^2$     | R-2D3D  | 78.1 | 41.2  
3D-RotNet       | 2018   | K400   | $16\times112^2$     | R3D-18  | 62.9 | 33.7  
ST-Puzzle        | 2019   | K400   | $16\times112^2$     | R3D-18  | 65.8 | 33.7   
DPC             | 2019   | K400   | $16\times128^2$     | R3D-18  | 68.2 | 34.5  
RTT           | 2020   | UCF   | $16\times112^2$     | R3D-18  | 77.3 | 47.5  
RTT           | 2020   | K400   | $16\times112^2$     | R3D-18  | 79.3 | **49.8**  
**PCL (3DRotNet)** | | UCF    | $16\times112^2$     | R3D-18  | 82.8 | 47.2  
**PCL (VCP)** | | UCF         | $16\times112^2$     | R3D-18  | 83.4 | 48.8  
**PCL (VCP)** | | K400         | $16\times112^2$     | R3D-18  | **85.6** | 48.0  
VCOP              | 2019   | UCF    | $16\times112^2$     | R3D     | 64.9 | 29.5  
VCP             | 2020   | UCF    | $16\times112^2$     | R3D     | 66.0 | 31.5  
PRP             | 2020   | UCF    | $16\times112^2$     | R3D     | 66.5 | 29.7  
IIC    | 2020   | UCF    | $16\times112^2$     | R3D     | 74.4 | 38.3  
**PCL (VCOP)** | | UCF        | $16\times112^2$     | R3D     | 78.2 | 40.5
**PCL (VCP)** | | UCF         | $16\times112^2$     | R3D     | **81.1** | **45.0**  
VCOP              | 2019   | UCF    | $16\times112^2$     | C3D     | 65.6 | 28.4  
VCP             | 2020   | UCF    | $16\times112^2$     | C3D     | 68.5 | 32.5  
PRP             | 2020   | UCF    | $16\times112^2$     | C3D     | 69.1 | 34.5  
RTT           | 2020   | K400   | $16\times112^2$     | C3D     | 69.9 | 39.6  
**PCL (VCOP)** | | UCF        | $16\times112^2$     | C3D     | 79.8 | 41.8  
**PCL (VCP)** | | UCF         | $16\times112^2$     | C3D     | **81.4** | **45.2**  
VCOP              | 2019   | UCF    | $16\times112^2$     | R(2+1)D | 72.4 | 30.9  
VCP             | 2020   | UCF    | $16\times112^2$     | R(2+1)D | 66.3 | 32.2  
PRP             | 2020   | UCF    | $16\times112^2$     | R(2+1)D | 72.1 | 35.0  
RTT           | 2020   | UCF    | $16\times112^2$     | R(2+1)D | 81.6 | 46.4  
PacePred        | 2020   | UCF    | $16\times112^2$     | R(2+1)D | 75.9 | 35.9  
PacePred        | 2020   | K400   | $16\times112^2$     | R(2+1)D | 77.1 | 36.6  
**PCL (VCOP)** | | UCF         | $16\times112^2$     | R(2+1)D     | 79.2 | 41.6  
**PCL (VCP)** | | UCF         | $16\times112^2$     | R(2+1)D     | 79.9 | 45.6  
**PCL (VCP)** | | K400         | $16\times112^2$     | R(2+1)D     | **85.7** | **47.4**  


## Citation
If you find our work helpful for your research, please consider citing the paper
```
@article{tao2021pcl,
    title={Pretext-Contrastive Learning: Toward Good Practices in Self-supervised Video Representation Leaning},
    author={Li Tao and Xueting Wang and Toshihiko Yamasaki},
    journal={arXiv preprint arXiv:xxx},
    year={2021},
    eprint={xxx},
}
```

If you find the residual input helpful for video-related tasks, please consider citing the paper
```
@article{tao2020rethinking,
  title={Rethinking Motion Representation: Residual Frames with 3D ConvNets for Better Action Recognition},
  author={Tao, Li and Wang, Xueting and Yamasaki, Toshihiko},
  journal={arXiv preprint arXiv:2001.05661},
  year={2020}
}

@inproceedings{tao2020motion,
  title={Motion Representation Using Residual Frames with 3D CNN},
  author={Tao, Li and Wang, Xueting and Yamasaki, Toshihiko},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  pages={1786--1790},
  year={2020},
  organization={IEEE}
}
```

## Acknowledgements
Part of this code is reused from [IIC](https://github.com/BestJuly/IIC) and [VCOP](https://github.com/xudejing/video-clip-order-prediction).