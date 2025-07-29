# GitHub Copilot Instructions for AgroAI Co-Pilot Project

## 1. Core Project Objective
The goal is to build a research prototype of an AI mobile app for Sri Lankan farmers. The system has four core AI modules providing real-time diagnosis, environmental stress analysis, graph-based advisory, and RL-based crop planning.

## 2. Technology Stack
- **Frontend:** Flutter
- **Backend:** Firebase (Firestore, Cloud Functions in TypeScript)
- **On-Device AI:** TensorFlow Lite (TFLite)
- **Cloud AI Training:** TensorFlow, PyTorch
- **Knowledge Graph:** Neo4j, spaCy, Cypher
- **Vector Search:** FAISS
- **LLM:** Gemma 2B
- **RL:** Gymnasium, PPO
- **Deployment:** Vertex AI for cloud models

## 3. Monorepo Structure & Workflow
This is a monorepo containing all project components.
- `mobile-app/`: Flutter frontend.
- `backend/`: Firebase Cloud Functions (TypeScript).
- `module1-edge-ai/`: Python project for the Disease Recognition Engine.
- `module2-knowledge-graph/`: Python project for the Graph Advisory System.
- `module3-multimodal-fusion/`: Python project for the Fusion Transformer.
- `module4-rl-optimization/`: Python project for the RL Optimization agent.

**CRITICAL WORKFLOW:** Most ML training is done in Google Colab to save local compute.
- `notebooks/` in each module contains `.ipynb` files for exploration and training. These notebooks ARE version controlled.
- `src/` contains reusable Python code (.py files) refactored from notebooks.
- `data/` and `trained_models/` directories are EXCLUDED by `.gitignore`. The actual data and large model files reside in Google Drive or Google Cloud Storage and are mounted/downloaded in Colab at runtime.
- `scripts/` contains scripts to run full pipelines.

## 4. Coding & Commit Style
- **Code:** Follow standard Python (PEP 8) and TypeScript best practices.
- **Commits:** Use Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
