# An Integrated AI Co-Pilot for Precision Agriculture: A Graph-Powered, Multi-Modal Approach to Optimizing Sri Lankan Farming

## Project Overview

Welcome to the repository for the **AgroAI Co-Pilot** project! This initiative, undertaken by a research team at the University of Moratuwa, aims to revolutionize Sri Lankan agriculture by developing an intelligent AI mobile platform. Our vision is to replace agricultural uncertainty with data-driven confidence, creating a trusted partner for farmers that is accessible on standard smartphones.

The core objective is to design, build, and validate a research prototype that provides real-time diagnostics, predictive environmental insights, and economically optimized strategic advice tailored for Sri Lankan farming conditions.

## Table of Contents

1.  [Vision & Objectives](#vision--objectives)
2.  [System Architecture & Technology Stack](#system-architecture--technology-stack)
3.  [Core Modules](#core-modules)
    * [Module 1: Calibrated & Class-Aware Disease Recognition Engine (Edge AI Specialist)](#module-1-calibrated--class-aware-disease-recognition-engine-edge-ai-specialist)
    * [Module 2: Auditable Graph-Powered Advisory System (Knowledge & Reasoning Engineer)](#module-2-auditable-graph-powered-advisory-system-knowledge--reasoning-engineer)
    * [Module 3: Quantitative Multi-Modal Fusion Transformer (Data Scientist & Fusion Modeler)](#module-3-quantitative-multi-modal-fusion-transformer-data-scientist--fusion-modeler)
    * [Module 4: Reinforcement Learning for Crop ROI Optimization (Research Scientist & Optimization Specialist)](#module-4-reinforcement-learning-for-crop-roi-optimization-research-scientist--optimization-specialist)
4.  [Operational Flow](#operational-flow)
5.  [Project Scope & Delimitations](#project-scope--delimitations)
6.  [Setup and Contribution Guide](#setup-and-contribution-guide)
    * [Repository Structure](#repository-structure)
    * [Local Development Setup](#local-development-setup)
    * [Contribution Guidelines](#contribution-guidelines)
7.  [License](#license)
8.  [Contact](#contact)

---

## Vision & Objectives

**Vision:** To replace agricultural uncertainty with data-driven confidence by creating an intelligent partner that is accessible on standard smartphones and trusted through transparent, reliable AI.

**Objective:** To design, build, and validate a research prototype of an AI mobile platform that provides Sri Lankan farmers with real-time diagnostics, predictive environmental insights, and economically optimized strategic advice.

---

## System Architecture & Technology Stack

Our system is designed for scalability, performance, and accessibility.

* **Frontend:** Cross-platform mobile application developed with **Flutter**.
* **Backend:** Serverless backend leveraging **Firebase** (Firestore for data, Cloud Functions for orchestration).
* **On-Device AI:** **TensorFlow Lite (TFLite)** for lightweight, real-time, and offline inference.
* **Cloud AI:** **PyTorch** and **TensorFlow** for computationally intensive model development and training on cloud infrastructure.
* **Knowledge Graph:** **Neo4j** for structured agricultural knowledge representation.
* **Vector Search:** **FAISS** for efficient semantic retrieval.

---

## Core Modules

The project is structured around four distinct, yet interconnected, research modules, each addressing a critical aspect of agricultural intelligence.

### Module 1: Calibrated & Class-Aware Disease Recognition Engine (Edge AI Specialist)

* **Lead:** Member 1
* **Research Gap:** Developing a class-aware quantization methodology to preserve accuracy for rare disease classes and ensure well-calibrated model confidence scores.
* **Methodology:**
    * Utilize **EfficientNetV2-B0** as the base architecture.
    * Implement **Quantization-Aware Training (QAT)** with a **Weighted Focal Loss** function to prioritize rare classes.
    * Apply **Temperature Scaling** post-QAT to optimize calibration by minimizing Expected Calibration Error (ECE).
* **Output:** A fully integer **INT8 TensorFlow Lite model** for on-device deployment.

### Module 2: Auditable Graph-Powered Advisory System (Knowledge & Reasoning Engineer)

* **Lead:** Member 2
* **Research Gap:** Pioneering a **Graph-Augmented, Self-Correcting Retrieval (GASR)** framework that provides verifiable database queries (e.g., Cypher) for full auditability.
* **Methodology:**
    * **Knowledge Graph Construction:** Use **spaCy** for NLP to extract entities and relationships from agricultural texts, ingested into a **Neo4j** graph database.
    * **Hybrid Retrieval:** Parallel vector search (FAISS) and structured graph search (Cypher).
    * **Critique-and-Refine Loop:** An agentic LLM (**Gemma 2B**) autonomously generates new, more specific Cypher queries to fill information gaps.
* **Output:** Natural language answers complemented by the last successful Cypher query for transparency.

### Module 3: Quantitative Multi-Modal Fusion Transformer (Data Scientist & Fusion Modeler)

* **Lead:** Member 3
* **Research Gap:** Creating a lightweight and interpretable multi-modal fusion transformer whose attention mechanisms can quantify the contribution of each data modality.
* **Methodology:**
    * **Architecture:** Custom transformer with separate embedding pathways for **Image Features** (from Module 1's backbone), **Time-Series Weather Data** (via GRU), and **Static Soil Properties** (via MLP).
    * **Fusion:** Self-attention and cross-attention layers combine embedded vectors.
    * **Attribution Score:** Aggregated and normalized attention weights provide a quantitative score (e.g., Weather: 71%, Soil: 22%, Image: 7%).
* **Output:** A predicted 'Crop Stress Score' (0-1) and the structured environmental attribution score.

### Module 4: Reinforcement Learning for Crop ROI Optimization (Research Scientist & Optimization Specialist)

* **Lead:** Member 4
* **Research Gap:** Formulating agricultural planning as a multi-objective deep reinforcement learning (MO-DRL) problem to learn dynamic policies for yield, profitability, and systemic risk trade-offs.
* **Methodology:**
    * **MDP Formulation:** State includes location, climate forecast, and market forecast; Action is the discrete crop choice.
    * **Multi-Objective Reward:** A weighted sum: $R = w_1 \cdot R_{yield} + w_2 \cdot R_{profit} - w_3 \cdot R_{risk}$.
    * **Training:** A **Proximal Policy Optimization (PPO)** agent trained in a custom **Gymnasium** environment using 20 years of historical Sri Lankan data.
* **Output:** Strategic crop recommendations based on farmer-defined risk preferences.

---

## Operational Flow

The AgroAI Co-Pilot integrates these modules into a seamless operational flow for farmers:

1.  **On-Device Scan:** A farmer takes a photo. **Module 1** provides an instant, calibrated disease diagnosis offline using the on-device TFLite model.
2.  **Cloud Enrichment:** The app uploads relevant data. **Module 3** fuses it with real-time weather and soil data to predict environmental stress and identify its root cause.
3.  **Intelligent Advice:** All gathered context is sent to **Module 2**, which performs its self-correcting reasoning and generates a deep, auditable advisory.
4.  **Strategic Planning (Separate Tool):** For long-term decisions, a farmer can use **Module 4** to receive a strategic crop recommendation for the next season, optimized for their specific risk preference.

---

## Project Scope & Delimitations

To maintain focus and ensure a deliverable prototype within the 6-month timeframe, the project has specific delimitations:

* **Language:** Core AI models are developed and validated exclusively in English. Localization for local languages will be handled by a third-party API translation layer in the final prototype, considered a UI feature.
* **Hardware:** The project does not involve the design or development of custom IoT hardware. The system is designed to consume data from existing weather APIs and public soil databases.

---

## Setup and Contribution Guide

This section outlines how to set up your development environment and contribute to the project.

### Repository Structure
```
agro-ai-copilot/
├── README.md
├── .github/             # GitHub Actions CI/CD workflows
│   └── workflows/
├── mobile-app/          # Flutter frontend
├── backend/             # Firebase functions, cloud AI deployment scripts
├── module1-edge-ai/     # Member 1's model training, TFLite conversion, evaluation
├── module2-knowledge-graph/ # Member 2's knowledge graph construction, retrieval logic
├── module3-multimodal-fusion/ # Member 3's fusion model development, data processing
├── module4-rl-optimization/ # Member 4's MDP formulation, RL agent training
└── documentation/       # Shared project documentation, research notes, reports
```

## Local Development Setup

To get started, ensure you have Git installed.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/UOM-AgroAI-Project/agro-ai-copilot.git](https://github.com/UOM-AgroAI-Project/agro-ai-copilot.git)
    cd agro-ai-copilot
    ```

2.  **Flutter (Mobile App - `mobile-app/`)**
    * Install [Flutter SDK](https://flutter.dev/docs/get-started/install).
    * Navigate to the `mobile-app` directory:
        ```bash
        cd mobile-app
        flutter pub get
        flutter run # To run on connected device/emulator
        ```

3.  **Firebase Backend (`backend/`)**
    * Install [Node.js (LTS version)](https://nodejs.org/en/download/) and [Firebase CLI](https://firebase.google.com/docs/cli).
    * Navigate to the `backend` directory:
        ```bash
        cd backend
        npm install
        firebase emulators:start # For local development and testing Firebase functions
        ```

4.  **Python-based AI Modules (`module1-edge-ai/`, `module2-knowledge-graph/`, `module3-multimodal-fusion/`, `module4-rl-optimization/`)**
    * For each module, it is **highly recommended** to use a Python virtual environment or Docker for dependency management.
    * **Using `venv` (recommended for most local dev):**
        ```bash
        cd moduleX-your-module/ # e.g., cd module1-edge-ai/
        python -m venv .venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        pip install -r requirements.txt
        # Run your module-specific scripts/tests
        ```
    * **Using Docker (recommended for cloud-based modules, e.g., Module 2, 3, 4):**
        * Ensure Docker Desktop is installed and running.
        * Each module will have a `Dockerfile`. Build and run it:
            ```bash
            cd moduleX-your-module/
            docker build -t agroai-moduleX .
            docker run -it agroai-moduleX bash # To enter the container environment
            ```

### Contribution Guidelines

We follow a **GitHub Flow** branching strategy:

1.  **`main` Branch:** This branch contains the stable, production-ready code.
2.  **Feature Branches:** For any new feature, bug fix, or research task, create a new branch from `main`:
    ```bash
    git checkout main
    git pull origin main
    git checkout -b feature/your-feature-name # e.g., feature/module1-qat-implementation
    ```
3.  **Develop:** Write your code, commit frequently with clear messages.
4.  **Pull Requests (PRs):** When your work is ready for review, push your branch and open a Pull Request against the `main` branch.
    * Ensure your code adheres to linting and formatting standards (automated by CI/CD).
    * Describe your changes clearly in the PR description.
    * Request reviews from relevant team members.
5.  **Code Review:** All PRs must be reviewed and approved by at least one other team member before merging.
6.  **CI/CD:** Our GitHub Actions workflows will automatically run tests and checks on every push and PR to ensure code quality and prevent regressions.

---

## License

This project is open-source under the [MIT License](LICENSE).

---

## Contact

For any inquiries, please contact the project leads or the research team at the University of Moratuwa.

* [Team Lead Email]
* [University Department Link/Email]