# mmdetection: 
MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project. With it, many common detector models can be implemented simply and efficiently, requiring only modification of some configuration files.

[mmdetction doc 3.x(latest) ](https://mmdetection.readthedocs.io/en/latest/user_guides/config.html)

[mmdetction doc 2.x(stable) ](https://mmdetection.readthedocs.io/en/stable/)

[mmdetction doc 2.x(2.28.2) ](https://mmdetection.readthedocs.io/en/v2.28.2/)

[mmdetction doc release tags ](https://github.com/open-mmlab/mmdetection/tags)


# Introduction for mmdetection part
Use the faster-RCNN model to locate the abnormal bounding box in ChestXray. Using dataset Vindr_CXR with instance-level labels.

Quoted from [the source code of mmdetction doc 2.x](https://github.com/open-mmlab/mmdetection/tree/2.x)

Introduction of COCO format [guides of COCO dataset](https://mmdetection.readthedocs.io/en/3.x/advanced_guides/customize_dataset.html) *This dataset format is different from yolo format


# File structure
````
mmdetection
 ├─ .github
 ├─ .gitignore
 ├─ Mytrain                               # **ALL Launch are here**
 │  ├─ Example_Launch_name1               # Experiment name
 │  │  └─                                 # Log and weights for this Experiment name
 │  ├─ Example_Launch_name2
 │  ├─ Example_Launch_name3
 │  ├─ config_Lauch_name1.py
 │  ├─ config_Lauch_name2.py
 │  ├─ config_Lauch_name3.py              # **Customized training config**
 │  └─ etc ...
 ├─ configs                               # **Default training config**
 │  ├─ models
 │  └─ mmdet
 │     └─ models
 │        └─ backbones
 │        │  ├─ __init__.py
 │        │  ├─ convnext                  # ALL backbones are here
 │        │  └─ etc ..
 │        └─ detectors                    # ALL detectors are here
 │        │  ├─ __init__.py
 │        │  ├─ faster-rcnn                    
 │        │  └─ etc ..
 ├─ demo
 │  ├─ MMDet_InstanceSeg_Tutorial.ipynb
 │  ├─ MMDet_Tutorial.ipynb
 │  ├─ create_result_gif.py
 │  ├─ image_demo.py
 │  ├─ inference_demo.ipynb
 │  ├─ video_demo.py
 │  ├─ video_gpuaccel_demo.py
 │  ├─ vis_cam.py                         # **Generate heatmap**
 │  └─ webcam_demo.py                     # **Generate heatmap**
 ├─ tools
 │  ├─ analysis_tools                     # **analysis tools**
 │  │  └─ ...
 │  ├─ test.py
 │  ├─ train.py
 │  └─ ...
````

# Getting Started on Windows and Linux
## Install on Windows
- [Visual Studio Code](https://code.visualstudio.com/) (optional)
- [Miniconda3 for windows](https://docs.conda.io/en/main/miniconda.html) (optional)

## Set enviroment
- Create env_mm2x enviroment

- check cuda version:
```
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
nvidia-smi
>Got
CUDA Version: 12.0
```
- can use cuda 11.3, for mmdet2.x need cuda11.7 + torch1.13.x + mmcv 1.7.0
- install torch1.13.x from https://pytorch.org/get-started/previous-versions/
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
- Install MMCV using MIM.
- Install components for open mmlab
```
pip install -U openmim
mim install mmengine
```
- Install mmcv 1.7.0
```
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```
- Install MMDetection
```
# git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

- [Verify the installation](https://mmdetection.readthedocs.io/en/v2.28.2/get_started.html) 

# ReBuild a Experiment
```
cd mmdetection
CUDA_VISIBLE_DEVICES=no_gpu nohup python tools/train.py config_doc > LOG_tmp.txt &
EX:
CUDA_VISIBLE_DEVICES=1 nohup python tools/train.py Mytrain/config_vindr_convnext.py > LOG_tmp.txt &
```

## results
There will create a new <Name> folder in the directory Mytrain, every time the program is lauching, the result and checkpoint will be saved in it.

You can alse use Weight and bias as the Developer tools for ML. Once logged into wandb， the results will be update at wandb.(including weights of models)

# HOW Evaluate a Experiment
Get result:
```
python tools/test.py config_NAME.py Model_weight_corresponding_config.pth --eval bbox --work-dir Path_to_save_pklresult
```
Get evaluation, Ex: Confusion Matrix:
```
python tools/analysis_tools/confusion_matrix.py config_NAME.py Path_to_save_pklresult eval/
```
For more detailed operations, can directly refer to the mmdetection documentation.

# Note
MMDetection is an object detection toolbox that contains a rich set of object detection, instance segmentation, and panoptic segmentation methods as well as related components and modules.

It is powerful and has many parameters available, but because it uses hooks to connect components, additional custom extensions require custom hooks to implement, which is difficult.
