# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Greenhouse plant disease classification using lightweight/compact CNNs. This is a Master's-level research project comparing three architectures — **MobileNetV4**, **ConvNeXt-V2 Nano**, and **GhostNetV3** — on plant disease image data, targeting deployment on edge/IoT devices.

**Language:** Vietnamese research project; code and variable names should use English, comments may be in Vietnamese.

**Runtime environment:** Kaggle / Google Colab Free Tier (T4 GPU, 16GB VRAM).

## Key Technical Decisions

- **Framework:** PyTorch with `timm` (PyTorch Image Models) for model creation
  - `timm.create_model('mobilenetv4_conv_small', pretrained=True)`
  - `timm.create_model('convnextv2_nano', pretrained=True)`
  - GhostNetV3: from official repo or `timm` if available
- **Preprocessing:** CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle greenhouse fog/LED lighting
- **Image size:** 224x224
- **Optimizer:** AdamW (weight_decay=0.05)
- **Loss:** Label Smoothing Cross-Entropy (alpha=0.1)
- **Scheduler:** Cosine Annealing with Warm Restarts
- **Precision:** Mixed Precision (FP16) for memory efficiency
- **Batch size:** 32 (or 64 with FP16)
- **Epochs:** 50

## Evaluation Metrics

All models must be compared across these metrics (not just accuracy):
- Top-1 Accuracy (target >92%)
- Macro F1-Score (target >88%) — primary metric due to class imbalance
- Inference Latency (target <30ms on T4, batch_size=1)
- Throughput (target >100 img/s)
- Model Size (target <10MB .pth/.onnx)
- FLOPs (target <1.0 GFLOPs)

## Required Outputs

- Quantitative comparison table across all 3 architectures
- Learning curves (Train/Val Loss and Accuracy over 50 epochs)
- Confusion matrices per model
- Accuracy vs. Latency scatter plot (trade-off analysis)
- Ablation study results (e.g., UIB module on/off for MobileNetV4, MixUp on/off for ConvNeXt-V2)
