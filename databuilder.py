# %%
# This cell should always be run first
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import numpy as np
from tqdm import tqdm
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=device)

# %%
# Creation of the padding token
pad = t2vec_model.predict(['End of text.'], source_lang="eng_Latn")[0]
np.save('/data/pad.npy', pad.numpy())

# %%
def pad_tensor(tensor, max_len=20): #truncate and pad sequences
    seq = tensor.size(0)
    if seq < max_len:
        padding_needed = max_len - seq
        padding = pad.expand(padding_needed, -1).to(device)
        tensor = torch.cat([tensor, padding], dim=0)
    return tensor[:max_len]

# %%
dataset = torch.load('fineweb_random.pth') #it's load_dataset("nickypro/fineweb-llama3b-regen-split")['train']['split_text'] that I shuffled

for i in tqdm([':100', '100:200', '200:300', '300:400', '400:500', '500:600', '600:700', '700:800', '800:900', '900:']):
    data = torch.load('fineweb['+str(i)+'].pth')
    
    if i==':100':
        n='100'
    elif i=='900:':
        n='900'
        data=data[:-2000] #we keep 1000 for the validation set and 1000 for the test set
    else:
        n=i[-3:]

    tensor_list_padded = torch.stack([pad_tensor(tensor.to(device)) for tensor in data])

    np.save('/data/'+n+'k.npy', tensor_list_padded.numpy())

# %% [markdown]
# Validation and Test Data:

# %%
dataset = torch.load('fineweb[900:].pth')

validation_set = torch.stack([pad_tensor(tensor.to(device)) for tensor in dataset[-2000:-1000]])
np.save('/data/val.npy', validation_set.numpy())


test_sonar = dataset[-1000:]

test_sonar = [sous_liste[:20] for sous_liste in test_sonar] #padding is not usefull in the way we test

prompt = []
output = []

for sample in test_sonar:
    prompt.append(sample[0])
    output.append(sample[1:])
    
prompt = torch.stack(prompt).unsqueeze(1)

torch.save([prompt, output], '/workspace/eloise/sentemb/data/test_sonarprompt_sonaroutput.pth')


