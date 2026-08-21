# FS-LCMF-Robot-Fault-Diagnosis
Explainable cross-condition fault diagnosis for industrial robots using reasoning-chain-guided multimodal fusion.
FS-LCMF-Robot-Fault-Diagnosis/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── paper/
│   └── manuscript.pdf
│
├── figures/
│   ├── framework.png
│   ├── confusion_matrix.png
│   ├── tsne_visualization.png
│   ├── dynamic_kg_initial.png
│   └── dynamic_kg_updated.png
│
├── sample_data/
│   └── normal_condition/
│       ├── C0S1L2_sample.npy
│       ├── C0S2L2_sample.npy
│       └── C0S3L2_sample.npy
│
├── results/
│   ├── cross_condition_results.csv
│   ├── ablation_results.csv
│   └── qwen_reasoning_results.csv
│
└── src/
    ├── data_preprocessing/
    ├── multimodal_fusion/
    ├── qwen_lora_reasoning/
    └── dynamic_knowledge_graph/

# Few-Shot Large-Model Reasoning-Chain-Guided Multimodal Fusion for Explainable Cross-Condition Fault Diagnosis of Industrial Robots

This repository provides the supplementary materials for our paper:

**Few-Shot Large-Model Reasoning-Chain-Guided Multimodal Fusion for Explainable Cross-Condition Fault Diagnosis of Industrial Robots**

The proposed framework, named **FS-LCMF**, integrates few-shot large-model reasoning-chain distillation, reasoning-generated-label-guided multimodal fusion, and dynamic knowledge graph explanation for cross-condition fault diagnosis of industrial robots.

## Overview

Industrial robots operate under variable speeds and loads, which may lead to significant distribution shifts in vibration signals, time-frequency patterns, and fault semantics. In practical applications, target-condition labels are usually unavailable, while directly invoking online large models for large-scale sample-by-sample inference is computationally expensive.

To address these issues, this work proposes a few-shot reasoning-chain-guided multimodal diagnosis framework. First, a small number of labeled source-condition samples are used to query an online teacher large model and construct diagnostic reasoning-chain supervision data. Then, the teacher-generated reasoning chains are distilled into a local Qwen-LoRA model, which generates reasoning-generated labels for large-scale unseen target-condition samples. Finally, vibration signals, CWT time-frequency images, and fault semantic embeddings are fused for cross-condition diagnosis, while a dynamic knowledge graph provides traceable diagnostic evidence.

## Main Features

- Few-shot diagnostic reasoning-chain construction using an online teacher large model.
- Qwen-LoRA-based reasoning-chain distillation for local target-condition inference.
- Multimodal fusion of vibration signals, CWT time-frequency images, and fault semantic embeddings.
- Reasoning-generated-label-guided semantic loading without using target-domain ground-truth labels.
- Dynamic knowledge graph explanation for traceable diagnostic evidence.
- Cross-condition diagnosis validation under different operating speeds.

## Repository Contents

```text
paper/          Manuscript or preprint file.
figures/        Main result figures, including framework, confusion matrices, t-SNE visualization, and knowledge graph visualization.
sample_data/    Partial publicly available sample data. Only normal-condition samples are provided for demonstration.
results/        Experimental result tables.
src/            Source code for preprocessing, reasoning-chain distillation, multimodal fusion, and dynamic knowledge graph construction.


    
