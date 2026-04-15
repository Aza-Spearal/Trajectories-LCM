# Using LCMs to Monitor LLMs

This repository contains the code associated with the following post:  
https://github.com/Aza-Spearal/lesswrong-posts_temporary/blob/main/using_lcms_to_monitor_llms.pdf

---

## 📌 Overview

This project explores the use of **Large Concept Models (LCMs)** to monitor and analyze the behavior of **Large Language Models (LLMs)**. It includes tools for dataset preparation, model training, and running control experiments.

---

## ⚙️ Workflow

### 1. Dataset Preparation

Use the `databuilder` script to generate the dataset required for training and evaluation.

---

### 2. Model Training

You can train different model variants using the following scripts. Each script launches a **Weights & Biases (wandb)** sweep for experiment tracking and hyperparameter optimization.

#### Available Training Scripts

- `llama_lora_wandb.py`  
  → Trains a **LLaMA model with LoRA (Low-Rank Adaptation)**

- `base_lcm_scratch_wandb.py`  
  → Trains a **re-implemented Base LCM from scratch**

- `base_lcm_full_wandb.py`  
  → Trains the **original Base LCM model**

---

### 3. Using the Original Base LCM

To run `base_lcm_full_wandb.py`, you must first install the official Large Concept Model (LCM) repository:

👉 https://github.com/facebookresearch/large_concept_model

#### Required Modifications

After cloning the repository, make the following changes:

1. **Replace architecture file**
   - Replace:
     ```
     large_concept_model/lcm/models/base_lcm/archs.py
     ```
   - With the version provided in this repository

2. **Modify sequence length**
   - Edit:
     ```
     large_concept_model/lcm/models/base_lcm/builder.py
     ```
   - Change:
     ```python
     max_seq_len: int = 2048
     ```
   - To:
     ```python
     max_seq_len: int = 20
     ```

---

## 🧪 Control Experiments

To reproduce the control experiments, simply run: control_experiments.ipynb


This notebook contains all necessary steps for evaluation and analysis.

---

## 📎 Notes

- Ensure that `wandb` is properly configured before launching training scripts.
- Some scripts may require significant computational resources depending on the model configuration.

---

## 🚀 Getting Started

1. Prepare the dataset using the `databuilder`
2. Choose a training script based on your setup
3. (Optional) Configure the original LCM repository if using the full model
4. Run control experiments for evaluation

---

## 📄 License

Refer to the respective repositories and dependencies for licensing details.
