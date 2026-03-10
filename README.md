# Vision Transformer Gesture Recognition for UAV Control

A deep learning based gesture recognition system that enables contactless UAV control using hand gestures and a Vision Transformer architecture.

The model is trained on a custom dataset and performs real-time gesture detection through webcam input.

This project implements a real time hand gesture recognition system using a Vision Transformer (ViT) architecture.

The goal is to enable contactless UAV control using natural hand gestures.

## Features

- Vision Transformer (ViT-B/16)
- Custom dataset
- PyTorch training pipeline
- Real-time webcam gesture detection


## Model Architecture

This project uses a **Vision Transformer (ViT-B/16)** architecture for gesture classification.

Unlike traditional Convolutional Neural Networks (CNNs), Vision Transformers process images as a sequence of patches and learn global relationships using self-attention mechanisms. This allows the model to capture long-range spatial dependencies and complex visual patterns that are useful for recognizing hand gestures.

The model is initialized with **ImageNet pretrained weights** and fine-tuned on a custom hand gesture dataset collected specifically for UAV control gestures.

Key details:

* Backbone: Vision Transformer (ViT-B/16)
* Input Resolution: 224 × 224
* Loss Function: CrossEntropyLoss
* Optimizer: AdamW
* Learning Rate Scheduler: ReduceLROnPlateau
* Framework: PyTorch


## Project Structure

gesture-controlled-uav
│
├── train_vit.py           # Vision Transformer training pipeline
├── webcam_inference.py    # Real-time gesture detection
├── requirements.txt       # Dependencies
└── README.md

## Installation

Clone the repository:

git clone https://github.com/sumedhaagh/gesture-controlled-uav.git

Install dependencies:

pip install -r requirements.txt


## Future Work

• Train additional gesture classes  
• Improve dataset diversity  
• Integrate gesture commands with UAV flight controller  
• Deploy optimized real-time inference model


## Current Gestures

Roll_up  
Throttle_up  

## Future Gestures

Stop  
Pitch_left  

## Training

Run:

python train_vit.py

## Inference

Run:

python webcam_inference.py


