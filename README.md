# GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models

This is the official implementation of the paper "GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models", accepted at NeurIPS 2024.

## Table of Contents
- [Code Explanation](#code-explanation)
- [Detailed Implementation for Models and Dataset](#detailed-implementation-for-models-and-dataset)
- [Evaluation Settings](#evaluation-settings)
- [Reference](#reference)

## Code Explanation

- `src/main.py`: Contains the function to run the main experiment of the paper and get the results.
- `src/figures_code/*`: Contains code to generate figures used in the paper. You can use the results to plot other graphs by substituting x and y values.
- `calibration.py`: Run this to perform calibration on CIFAR10 and obtain corresponding temperature constants.
- `src/face_code/request*.py`: Code for the online face API section. Run these files for each API and store prediction values in .npy files.
  - For DeepFace API, please refer to [DEEPFACE](https://github.com/serengil/deepface).
  - For other APIs, subscribe to the services, obtain your token, and insert it into the respective files.

For CELEB-HQ GAN, use [InterFACE-GAN](https://github.com/genforce/interfacegan) to control subgroup generation. Use the classifier mentioned in the paper to predict ground truth labels.

## Detailed Implementation for Models and Dataset

1. Create two directories named `samples` and `models` to store generated samples and robust models.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   This will install all necessary dependencies, including `robustbench` and `autoattack`.
3. To generate samples please refer to some GANs, we here provide the examples we used:
   - Use [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)
   - Download the GAN checkpoint
   - Generate samples with:
     ```
     CUDA_VISIBLE_DEVICES=0 python3 src/main.py -s -cfg PyTorch-StudioGAN/src/configs/CIFAR10/StyleGAN2-D2DCE-ADA.yaml -ckpt ckpt/ -metrics is --data_dir data/cifar10
     ```
   - Any dataset in .npz format containing 'x' (image features) and 'y' (labels) groups can be used.

## Evaluation Settings

Three sub-settings are available in the code:

1. `--lower_bound_eval_global`: Runs CW attack on 1000 images and reports average distortion. For true average distortion, store CW attack results per image as .npy files and ignore misclassified images.

2. `--lower_bound_eval_local`: Runs CW attack on 50 images and reports distortion for each image to compare with GREAT score.

3. `--robust_accuracy`: Runs AutoAttack on generated samples to report AutoAttack accuracy.

You can modify the sample size in each sub-setting to change the number of samples for evaluation.

## Reference

```
@inproceedings{
anonymous2024great,
title={{GREAT} Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models},
author={Anonymous},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=vunJCq9PwU}
}