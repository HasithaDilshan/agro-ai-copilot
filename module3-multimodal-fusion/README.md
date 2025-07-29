# Module 3: Quantitative Multi-Modal Fusion Transformer

## Objective & Research Gap
This module creates a lightweight, interpretable multi-modal fusion transformer. The key innovation is translating the model's attention mechanisms into a structured environmental attribution score, explicitly quantifying the contribution of each data modality (image, weather, soil).

## Methodology
1.  **Architecture:** A custom lightweight transformer with separate embedding pathways (Image Features, Time-Series Weather via GRU, Static Soil via MLP).
2.  **Fusion:** Process embedded vectors through self-attention and cross-attention layers.
3.  **Attribution Score:** Aggregate and normalize final cross-attention weights by modality to produce a quantitative score.
4.  **Output:** A 'Crop Stress Score' (0-1) and the structured attribution score.

## Evaluation Protocol
- **Quantitative:** Report Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) vs. a baseline XGBoost model.
- **Qualitative:** Case studies showing plausible attribution scores for different scenarios (e.g., drought).

## Local Folder Structure
- `notebooks/`: For data fusion experiments and model training.
- `src/`: For the transformer architecture, embedding layers, and data loaders.
- `scripts/`: To run training and evaluation pipelines.
- `data/`, `trained_models/`: Gitignored folders. Fused datasets and models live on cloud storage.
