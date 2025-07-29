# Module 2: Auditable Graph-Powered Advisory System

## Objective & Research Gap
This module pioneers a Graph-Augmented, Self-Correcting Retrieval (GASR) framework. The key innovation is making the reasoning process transparent by outputting the final database query (Cypher) used to generate the answer, ensuring full auditability.

## Methodology
1.  **Knowledge Graph:** Use spaCy to parse texts and ingest entities/relationships into Neo4j.
2.  **Hybrid Retrieval:** Parallel FAISS vector search and Cypher graph search.
3.  **Critique-and-Refine Loop:** A Gemma 2B agent assesses retrieved info. If incomplete, it generates a new Cypher query to fetch missing facts.
4.  **Output:** A natural language answer, appended with the final, successful Cypher query.

## Evaluation Protocol
- **Quantitative:** Compare GASR vs. baseline RAG on Factual Correctness (1-5 scale) and Answer Completeness (%).
- **Qualitative:** A case study diagram illustrating the self-correction loop.

## Local Folder Structure
- `notebooks/`: For KG construction, NLP experiments.
- `src/`: For graph ingestion logic, retrieval functions, agent definition.
- `scripts/`: To run the full retrieval and generation pipeline.
- `data/`, `trained_models/`: Gitignored folders. Raw texts, graph data, and FAISS indexes live on cloud storage.
