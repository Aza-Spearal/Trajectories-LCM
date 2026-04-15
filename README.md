Here is the code associated with the post [https://github.com/Aza-Spearal/lesswrong-posts_temporary/blob/main/using_lcms_to_monitor_llms.pdf]

### Workflow

**Dataset Preparation**  
Use the databuilder script to create the required dataset.

**Model Training**  
You can either use llama_lora_wandb.py, base_lcm_scratch_wandb.py, base_lcm_full_wandb.py to initiate a Weights & Biases (wandb) sweep and train model.

llama_lora_wandb.py correspond to Llama LoRA

base_lcm_scratch_wandb.py correspond to Re-implemented Base-LCM

base_lcm_full_wandb.py correspond to original Base-LCM model.

To execute the last one, you will need the Large Concept Model (LCM) repository, available at: https://github.com/facebookresearch/large_concept_model. You should modify 2 files in large_concept_model/lcm/models/base_lcm.
Replace the archs.py in the original LCM repository by the one provide here. You should also modify the line max_seq_len: int = 2048 by max_seq_len: int = 20 of the file builder.py.

### Control Experiemnts

You only have to run the control_experiments.ipynb file.
