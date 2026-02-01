# PINNs for Ekman Spiral Inverse Problem

This repository provides the official implementation of the Physics-Informed Neural Network (PINN) used in our paper to solve the inverse problem of the **Ekman Spiral**.

## 1. Project Overview
The core of this project is a PINN framework that integrates the governing equations of the Ekman layer into the neural network's loss function to identify fluid parameters.

## 2. Directory Structure
- `data/`: Directory containing the datasets (`u2.npz`, `v2.npz`).
- `train_pinn.py`: Main training and evaluation script.
- `requirements.txt`: Environment dependencies.

## 3. Requirements
This code requires **TensorFlow 1.15**. To install dependencies:
```bash
pip install tensorflow==1.15 numpy scipy matplotlib
