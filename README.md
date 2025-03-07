![2025-03-07 13_18_25-AN IMAGE IS WORTH 16X16 WORDS TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE pdf an](https://github.com/user-attachments/assets/1cfe07c5-c9c8-49a3-97e4-061a39a5126b)

## Overview

This repository contains an implementation of the Vision Transformer (ViT) model as described in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Alexey Dosovitskiy et al ([link to the paper](https://arxiv.org/abs/2010.11929)).

The Vision Transformer applies a standard Transformer architecture directly to sequences of image patches for the task of image classification, achieving competitive results on various benchmarks compared to state-of-the-art convolutional networks.

## Features

- Implementation of the Vision Transformer (ViT) model.
- Supports training and fine-tuning on various image classification datasets.
- Compatible with common machine learning frameworks such as TensorFlow or PyTorch.

## Requirements
To run the code, ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- PyTorch

## Usage
```
from vit import VisionTransformer

images = torch.randn(10, 3, 32, 32)
model = VisionTransformer(main_dim=512, patch_size=16, num_image_channels=3, num_classes=100)
model(images).shape
# (10, 100)
```

## Results

The Vision Transformer achieves the following results on popular benchmarks:

- ImageNet: 88.55% top-1 accuracy
- CIFAR-100: 94.55% top-1 accuracy

For more details, refer to the paper.


## Acknowledgments

- The original paper by Alexey Dosovitskiy et al.
- Google Research, Brain Team for their contributions to the Vision Transformer model.
