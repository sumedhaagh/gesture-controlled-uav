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
