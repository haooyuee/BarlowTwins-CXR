# BarlowTwins-CXR: Enhancing Chest X-Ray abnormality localization in heterogeneous data with cross-domain self-supervised learning

This repository contains
* A modified version of [the Barlowtwins paper](https://github.com/facebookresearch/barlowtwins) to accommodate for the NIH-CXR dataset and Linear Evaluation Protocol.
* A modified version of the original implementation of [mmDetection v.2.28.2](https://github.com/open-mmlab/mmdetection),adds code for visualizing heat maps.
 
A preprint of this work is available on [arXiv](https://arxiv.org/abs/2402.06499)
```
@article{sheng2024barlowtwinscxr,
      title={BarlowTwins-CXR: Enhancing Chest X-Ray abnormality localization in heterogeneous data with cross-domain self-supervised learning}, 
      author={Haoyue Sheng and Linrui Ma and Jean-Francois Samson and Dianbo Liu}
}
```

## Abstract
```
Background: Chest X-ray imaging-based abnormality localization, essential in diagnosing various diseases, faces significant clinical challenges due to complex interpretations and the growing workload of radiologists. While recent advances in deep learning offer promising solutions, there is still a critical issue of domain inconsistency in cross-domain transfer learning, which hampers the efficiency and accuracy of diagnostic processes. This study aims to address the domain inconsistency problem and improve autonomic abnormality localization performance of heterogeneous chest X-ray image analysis, by developing a self-supervised learning strategy called "BarlwoTwins-CXR".
Methods: We utilized two publicly available datasets: the NIH Chest X-ray Dataset and the VinDr-CXR. The BarlowTwins-CXR approach was conducted in a two-stage training process. Initially, self-supervised pre-training was performed using an adjusted Barlow Twins algorithm on the NIH dataset with a Resnet50 backbone pre-trained on ImageNet. This was followed by supervised fine-tuning on the VinDr-CXR dataset using Faster R-CNN with Feature Pyramid Network (FPN).
Results: Our experiments showed a significant improvement in model performance with BarlowTwins-CXR. The approach achieved a 3% increase in mAP50 accuracy compared to traditional ImageNet pre-trained models. In addition, the Ablation CAM method revealed enhanced precision in localizing chest abnormalities.
Conclusion: BarlowTwins-CXR significantly enhances the efficiency and accuracy of chest X-ray image-based abnormality localization, outperforming traditional transfer learning methods and effectively overcoming domain inconsistency in cross-domain scenarios. Our experiment results demonstrate the potential of using self-supervised learning to improve the generalizability of models in medical settings with limited amounts of heterogeneous data.
```

## Methods
BarlowTwins-CXR is a Dual-phase Training Framework. In Phase One, where Barlowtwins as a self-supervised pertaining method. The pairs of distorted images are processed through a shared ResNet network to produce embeddings. These are then compared using an empirical cross-correlation matrix ***C***, striving for the identity matrix ***I*** to minimize redundancy in feature dimensions, and optimizing the loss function ***L<sub>BT</sub>***. In Phase Two, the pre-trained ResNet backbone from Phase One is integrated into a Faster R-CNN architecture. It starts with multi-scale feature extraction through the Feature Pyramid Network (FPN), followed by the Region Proposal Network (RPN) that generates object region proposals. The features are then pooled and processed by fully connected (FC) layers to output the final class labels and bounding box coordinates for object detection tasks.

<img src="https://github.com/haooyuee/BarlowTwins-CXR/assets/104264477/fee2328e-8276-4340-ace6-f5d21713af8b" alt="drawing" width="550"/>

## Evaluation
Comparison of BarlowTwins-CXR performance against ResNet initialized baseline for abnormality localization on the VinDr-CXR dataset.

<img src="https://github.com/haooyuee/BarlowTwins-CXR/assets/104264477/c6711d97-33b0-4f40-908e-c76306876c07" alt="drawing" width="550"/>

Comparison of Heatmaps of BarlowTwins-CXR performance against ResNet initialized baseline on the training and test sets.

<img src="https://github.com/haooyuee/BarlowTwins-CXR/assets/104264477/02241b41-2a5c-401e-a807-89ce00a0f949" alt="drawing" width="750"/>

Comparison of BarlowTwins-CXR performance against ResNet initialized baseline when only the linear layers are fine-tuned. (Linear Evaluation Protocol)

<img src="https://github.com/haooyuee/BarlowTwins-CXR/assets/104264477/307a7df3-f29e-41ef-883c-8dbf775f10b7" alt="drawing" width="350"/>

Comparison of BarlowTwins-CXR performance against ResNet initialized baseline when all the linear layers are fine-tuned. (End-to-End Finetuning)

<img src="https://github.com/haooyuee/BarlowTwins-CXR/assets/104264477/1644a58a-9ca4-4fc5-829f-2f8682cef681" alt="drawing" width="350"/>

## Checkpoints
coming soon ...

## Running the experiments
Our experiments were conducted using the servers of the Center intégré universitaire de santé et de services sociaux du Centre-Sud-de-l'Île-de-Montréal.
### Pre-Training
The specific steps of the Barlowtwins-CXR pre-training process can be found [here](https://github.com/facebookresearch/barlowtwins), as well as the comments at the top of the code.

You also need to download the NIH-CXR dataset, which can be downloaded from [here](https://www.kaggle.com/datasets/nih-chest-xrays/data).

### BarlowTwins-CXR Training with Vindr-CXR
The specific steps of the Barlowtwins-CXR training on the Vindr-CXR dataset can be found [here](https://github.com/haooyuee/BarlowTwins-CXR/blob/main/mmdetection/README.md), as well as the official [documentation](https://mmdetection.readthedocs.io/en/latest/user_guides/config.html) of mmdetection.

You need to download the Vindr-CXR dataset, which can be downloaded from [here](https://physionet.org/content/vindr-cxr/1.0.0/) or from [Kaggle](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data)

Regarding the WBF and training set partitioning in preprocessing, you can reproduce it in this notebook.

## Additional Information
[NIH-CXR dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)

[Vindr-CXR dataset](https://vindr.ai/datasets/cxr)

[mmdetection](https://github.com/open-mmlab/mmdetection)

[Barlowtwins](https://github.com/facebookresearch/barlowtwins)

[Heatmap for mmdetection](https://github.com/open-mmlab/mmdetection/pull/7987)


