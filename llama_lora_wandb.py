from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gc
import wandb
import numpy as np
from itertools import count
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_emb = 1024

pad = torch.from_numpy(np.load('../data/pad.npy')).to(device)

cos = nn.CosineSimilarity(dim=-1)
cossim = nn.CosineEmbeddingLoss()
mse = nn.MSELoss()


def criterion(output, target):
    mask = ~(target == pad).all(dim=-1)  # [batch, seq]
    target = target[mask]
    output = output[mask]

    cossim_ = cossim(output, target, torch.full((1,), 1).to(device))
    mse_ = mse(output, target)
    return [cossim_, mse_]


def validation(epoch, cpt, model, valloader, val_list):
    cossim_loss = 0
    mse_loss = 0
    model.eval()
    with torch.no_grad():
        for src in valloader:
            src = src.to(device)
            outputs = model(src[:, :-1])
            [cossim_, mse_] = criterion(outputs, src[:, 1:])
            cossim_loss += cossim_.item()
            mse_loss += mse_.item()
        cossim_loss = cossim_loss / len(valloader)
        mse_loss = mse_loss / len(valloader)
        val_list.append([cossim_loss, mse_loss])
        print('Epoch', epoch + 1, 'Part', cpt, "Cossim", cossim_loss, "MSE", mse_loss)
        wandb.log({"Cossim": cossim_loss, "MSE": mse_loss, "epoch": epoch + 1, "part": cpt})
    model.train()
    return val_list, cpt + 1


def autoregr_infer(model, src):
    seq = 20 - 1
    list_autoregr = []
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        output = model(src)
        autoregr = torch.cat((src, output), dim=1)
        for i in range(seq - 1):
            out = model(autoregr)
            output = out[:, -1].unsqueeze(1)
            autoregr = torch.cat((autoregr, output), dim=1)
        list_autoregr.append(autoregr[:, 1:])
    return torch.cat(list_autoregr, dim=0)


def calculate_score(output, targets):
    seq = 20 - 1

    score_one_sum = 0
    score_sum_pad = torch.zeros(seq).to(device)
    pad_nbr_sum = torch.zeros(seq).to(device)

    for batch, target in enumerate(targets):
        out = output[batch][: len(target)].to(device)
        score = cos(out, target)
        score_one_sum += score.mean()
        score_sum_pad += F.pad(score, (0, seq - len(score)))
        pad_nbr_sum += F.pad(torch.ones(len(score)), (0, seq - len(score))).to(device)

    paragraphed_score = score_sum_pad / pad_nbr_sum
    paragraphed_score = [round(elem.item(), 2) for elem in paragraphed_score]
    final_score = score_one_sum.item() / len(targets)
    return final_score, paragraphed_score


def test(model, test_data):
    sonarprompt, sonaroutput = test_data

    output_autoregr = autoregr_infer(model, sonarprompt)
    final_sonar, paragraphed_sonar = calculate_score(output_autoregr, sonaroutput)

    print('Sonar score:', final_sonar, paragraphed_sonar)
    # ── W&B log ──────────────────────────────────────────────────────────────
    wandb.log({'Final sonar': final_sonar, 'Paragraphed sonar': str(paragraphed_sonar)})

    torch.cuda.empty_cache()
    del sonarprompt, sonaroutput, output_autoregr
    gc.collect()


class Model(nn.Module):
    def __init__(self, d_emb, original_model, config):
        super().__init__()
        hidden_size = original_model.config.hidden_size

        self.prenorm = nn.LayerNorm(d_emb)
        self.prelinear = nn.Linear(d_emb, hidden_size)
        self.postlinear = nn.Linear(hidden_size, d_emb)
        self.postnorm = nn.LayerNorm(d_emb)

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_rslora=True,
            target_modules='all-linear',
            r=config['r'],
            lora_alpha=config['r'] * config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            bias=config['lora_bias'],
        )
        self.mod = get_peft_model(original_model, lora_config)

    def forward(self, x):
        x = self.prenorm(x)
        x = self.prelinear(x)
        out = self.mod(inputs_embeds=x.to(torch.bfloat16), output_hidden_states=True)
        x = out.hidden_states[-1].to(torch.float32)
        x = self.postlinear(x)
        x = self.postnorm(x)
        return x

def check_for_nans(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in gradient of {name}")
            wandb.log({'Cossim': 10, 'MSE': 10})
            return True
        if torch.isnan(param.data).any():
            print(f"NaN detected in weights of {name}")
            wandb.log({'Cossim': 10, 'MSE': 10})
            return True
    return False


# ── Main training objective (called once per W&B sweep agent run) ─────────────
def objective(data):
    # Initialise the run and pull hyperparameters from wandb.config
    run = wandb.init()        # sweep agent already knows which project/sweep
    config = wandb.config     # populated automatically by the sweep agent

    orig_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        dtype=torch.bfloat16,
    )
    model = Model(d_emb, orig_model, config).to(device)

    optimizer = AdamW(
        [
            *model.prenorm.parameters(), *model.prelinear.parameters(),
            *model.mod.parameters(),
            *model.postlinear.parameters(), *model.postnorm.parameters(),
        ],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    model_save = model
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['T_0'])
    valloader = DataLoader(data[1], batch_size=64)

    val_list = []
    cpt = 1
    loss_type = 1   # COSSIM: 0 | MSE: 1
    patience = 3


    validations_per_epoch = 8

    print('')

    for epoch in count(0):
        print('Epoch', epoch)
        if epoch%10==0:
            epoch_traindata = torch.from_numpy(data[0])
        else:
            epoch_traindata = torch.from_numpy(np.load('/workspace/eloise/sentemb/data/'+str(epoch%10)+'00k.npy'))
            
        trainloader = DataLoader(epoch_traindata, config.batch_size)
        checker = len(trainloader) // validations_per_epoch

        for i, src in enumerate(trainloader):
            src = src.to(device)
            outputs = model(src[:, :-1])
            loss = criterion(outputs, src[:, 1:])[loss_type]
            loss.backward()

            if check_for_nans(model):
                print("Stopping training due to NaNs.")
                break

            optimizer.step()
            scheduler.step()

            wandb.log({"train_loss": loss.item()})

            if i != 0 and i % checker == 0:
                val_list, cpt = validation(epoch, cpt, model, valloader, val_list)

                if val_list[0][1] > 0.8:
                    break
                if cpt == 5 and val_list[-1][1] > 0.1:
                    break

                # Early stopping
                if len(val_list) >= 2:
                    loss_list = [x[loss_type] for x in val_list]
                    if loss_list[-1] < min(loss_list[:-1]):
                        model_save = copy.deepcopy(model)
                    elif (len(loss_list) - loss_list.index(min(loss_list))) > patience:
                        print(
                            'Best: Part', cpt - 1 - patience,
                            'Cosine Similarity', val_list[-1 - patience][0],
                            'MSE', val_list[-1 - patience][1],
                        )
                        wandb.log({
                            'Best Cossim': val_list[-1 - patience][0],
                            'Best MSE': val_list[-1 - patience][1],
                        })
                        model = model_save

                        if val_list[-1 - patience][1] < 5.75e-05:
                            test(model, data[2])

                        return



# ── W&B sweep configuration (mirrors the Ray Tune search space) ───────────────
sweep_config = {
    "method": "bayes",          # Bayesian optimisation ≈ HyperOptSearch
    "metric": {
        "name": "MSE",
        "goal": "minimize",
    },
    "early_terminate": {        # ASHA-style pruning
        "type": "hyperband",
        "min_iter": 3,
    },
    "parameters": {
        "r":             {"values": [4, 8, 16, 32, 64, 128, 256]},
        "lora_alpha":    {"min": 1, "max": 2},
        "lora_dropout":  {"min": 0.0, "max": 0.5},
        "lora_bias":     {"values": ["none", "all", "lora_only"]},
        "lr":            {"distribution": "log_uniform_values", "min": 1e-5, "max": 0.1},
        "weight_decay":  {"distribution": "log_uniform_values", "min": 1e-5, "max": 0.1},
        "T_0":           {"min": 10, "max": 100000},
        "batch_size":    {"values": [8, 16, 32, 64]},
    },
}

# ── Data loading ──────────────────────────────────────────────────────────────
np_data = np.load('../data/100k_1k_0.npz')
[test_prompt, test_output] = torch.load('/workspace/eloise/sentemb/data/test_sonarprompt_sonaroutput.pth')

data = [
    np_data['train'],
    np_data['val'],
    [test_prompt, test_output],
]
torch.cuda.empty_cache()
del np_data
gc.collect()

# ── Launch sweep ──────────────────────────────────────────────────────────────
sweep_id = wandb.sweep(sweep_config, project="Llama-lora_tune")

# num_samples=1000 → count=1000 in the agent
wandb.agent(sweep_id, function=lambda: objective(data), count=1000)