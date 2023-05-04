## Description
This is a PyTorch Re-Implementation of [EAST: An Efficient and Accurate Scene Text Detector](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf).

* ResNet50 as the feature extractor backbone.
* Feature Pyramid Network in the feature merging branch.
* ReduceLROnPlateau as the lr scheduler with 0.8 factor and 2000 patience.
* Only RBOX part is implemented.
* The pre-trained model provided achieves __82.12__ F-score on ICDAR 2015 Challenge 4 using only the 1000 images.

| Model | Recall | Precision | F-score | 
| - | - | - | - |
| ResNet50 + FPN | 80.48 | 85.31 | 82.12 |

## Prerequisites
Required libraries
* Anaconda 3
* PyTorch 1.10.1
* Lanms 1.0.2
* Shapely 1.8.0
* Opencv-python 4.5.4.60
* Pip 21.2.4
* Python 3.9.7
* Scipy 1.8.0
* Numpy 1.20.1
* Wandb
* Pillow 9.0.1
* Torchvision 0.11.2
* Importlib 4.8.1
* Matplotlib 3.4.3

## Installation
### 1. Clone the repo

```
git clone https://github.com/ChunFei96/EAST_resnet50.git
cd EAST_resnet50
```

### 2. Data & Pre-Trained Model
* Download Train and Test Data: [ICDAR 2015 Challenge 4](http://rrc.cvc.uab.es/?ch=4&com=downloads). Cut the data into four parts: train_img, train_gt, test_img, test_gt.

### 3. File structure
```
.
├── EAST_resnet50
│   ├── evaluate
│   ├── lanms
│   ├── pths
│   ├── torchvision
|   ├── dataset.py
|   ├── detect.py
|   ├── eval.py
|   ├── loss.py
|   ├── model.py
|   ├── resnet.py
|   ├── train.py
|   └── utils.py
└── ICDAR_2015
    ├── test_gt
    ├── test_img
    ├── train_gt
    └── train_img
```
## Train
Modify the parameters in ```train.py``` and run:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
## Detect
Modify the parameters in ```detect.py``` and run:
```
CUDA_VISIBLE_DEVICES=0 python detect.py
```
## Evaluate
* The evaluation scripts are from [ICDAR Offline evaluation](http://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) and have been modified to run successfully with Python 3.7.1.
* Change the ```evaluate/gt.zip``` if you test on other datasets.
* Modify the parameters in ```eval.py``` and run:
```
CUDA_VISIBLE_DEVICES=0 python eval.py
```

## Live metrics 
* wandb.ai is used to record and monitor live metrics such as Recall, Precision and F1-score.
* Please modify the parameter under the wandb init and config to direct to your wandb project name and entity in train.py.
```
wandb.init(project="xxx", entity="xxx")

wandb.config = {
        "pths_path": pths_path,
        "batch_size": batch_size,
        "lr": lr,
        "num_workers": num_workers,
        "epoch_iter": epoch_iter,
        "save_interval": save_interval,
    }
```


