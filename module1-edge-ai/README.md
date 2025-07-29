# Module 1: Calibrated & Class-Aware Disease Recognition Engine

## Objective & Research Gap
This module focuses on developing a highly accurate and reliable plant disease recognition model for on-device deployment. The key research challenge is to maintain high accuracy on rare disease classes during model compression (quantization) and to ensure the model's confidence scores are well-calibrated.

## Methodology
1.  **Architecture:** EfficientNetV2-B0.
2.  **Baseline:** Train a floating-point (FP32) model.
3.  **Quantization:** Implement Quantization-Aware Training (QAT) to produce an INT8 model.
4.  **Class Imbalance:** Use a **Weighted Focal Loss** during QAT.
5.  **Calibration:** Apply **Temperature Scaling** post-training to calibrate confidence scores by minimizing Expected Calibration Error (ECE).
6.  **Final Output:** A TensorFlow Lite (`.tflite`) model, quantized to INT8.

## Evaluation Protocol
- **Quantitative:** Compare FP32 vs. INT8 models on Top-1 Accuracy, F1-Score (5 rarest classes), Model Size (MB), Latency (ms), and ECE.
- **Qualitative:** A reliability diagram (calibration plot).

## Local Folder Structure
- `notebooks/`: Contains Colab notebooks (e.g., `fp32_baseline_training.ipynb`).
- `src/`: Reusable Python code (e.g., `losses.py`, `model_builder.py`).
- `scripts/`: Scripts for automated pipelines.
- `data/`, `trained_models/`: Gitignored folders. Data/models live on cloud storage.
