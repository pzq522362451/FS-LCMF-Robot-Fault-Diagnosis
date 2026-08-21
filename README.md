
This repository provides the supplementary materials for our paper:

**Few-Shot Large-Model Reasoning-Chain-Guided Multimodal Fusion for Explainable Cross-Condition Fault Diagnosis of Industrial Robots**

The proposed framework, named **FS-LCMF**, integrates few-shot large-model reasoning-chain distillation, reasoning-generated-label-guided multimodal fusion, and dynamic knowledge graph explanation for cross-condition fault diagnosis of industrial robots.

## Overview

The proposed framework, named **FS-LCMF**, integrates few-shot large-model reasoning-chain distillation, reasoning-generated-label-guided multimodal fusion, and dynamic knowledge graph explanation for cross-condition fault diagnosis of industrial robots.

Industrial robots operate under variable speeds and loads, which may lead to significant distribution shifts in vibration signals, time-frequency patterns, and fault semantics. In practical applications, target-condition labels are usually unavailable, while directly invoking online large models for large-scale sample-by-sample inference is computationally expensive.

To address these issues, this work proposes a few-shot reasoning-chain-guided multimodal diagnosis framework. First, a small number of labeled source-condition samples are used to query an online teacher large model and construct diagnostic reasoning-chain supervision data. Then, the teacher-generated reasoning chains are distilled into a local Qwen-LoRA model, which generates reasoning-generated labels for large-scale unseen target-condition samples. Finally, vibration signals, CWT time-frequency images, and fault semantic embeddings are fused for cross-condition diagnosis, while a dynamic knowledge graph provides traceable diagnostic evidence.

## Repository Contents

```text

figures/        Main result figures, including framework, confusion matrices, t-SNE visualization, and knowledge graph visualization.
sample_data/    Partial publicly available sample data. Only normal-condition samples are provided for demonstration.
results/        Experimental result tables.
src/            Source code for preprocessing, reasoning-chain distillation, multimodal fusion, and dynamic knowledge graph construction.


    
