Here is the code associated with the article [https://github.com/Aza-Spearal/lesswrong-posts_temporary/blob/main/using_lcms_to_monitor_llms.pdf]

To execute it, you will need a text dataset and the Large Concept Model (LCM) repository, available at: https://github.com/facebookresearch/large_concept_model.

Replace the builder.py and archs.py files with the corresponding versions from the large_concept_model/lcm/models/base_lcm/ directory.

### Workflow

**Dataset Preparation**  
Use the dataset_builder script to create the required dataset.

**Model Training**  
Run the LCM_sweep script to initiate a Weights & Biases (wandb) sweep using the LCM models.

**Evaluation**  
Use the evaluation script to assess the performance of a trained model.


### Evaluation Methods

We use two complementary evaluation approaches:

**Vector Similarity with SONAR**  
The model’s output is compared to the corresponding SONAR vector. However, while vectors may appear similar, the generated text could differ from the target text.

**Vector Similarity with Jasper**  
To address this, we decode the model’s output into text, then re-encode it using infgrad/jasper_en_vision_language_v1, a robust sentence embedding model. We compare this encoding to the Jasper embedding of the target text to better assess semantic similarity.
