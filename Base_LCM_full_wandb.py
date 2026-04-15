# %%
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gc
import numpy as np
import math
import datetime
import copy

import wandb
import traceback
import itertools

from transformers.utils import logging
logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_emb = 1024

pad = torch.from_numpy(np.load('/workspace/eloise/sentemb/data/pad.npy')).to(device)

cos = nn.CosineSimilarity(dim=-1)
cossim = nn.CosineEmbeddingLoss()

def check_for_nans(model):
    cpt=0
    for name, param in model.named_parameters():
        for attr_name, attr_value in [("weights", param.data), ("grad", param.grad)]:
            if param.grad is not None:
                if torch.isnan(attr_value).any():
                    print(f"NaN detected in {attr_name} of {name}", cpt)
                    cpt += 1
                    return True
    return False

# %%
from lcm.train.optim import build_lr_scheduler
from lcm.train.lcm.criterion import compute_standard_mse
from fairseq2.optim import DynamicLossScaler, AdamW
from fairseq2.gang import Gang
from lcm.datasets.batch import EmbeddingsBatch

def count_before_first_pad(tensor, pad):
    mask = torch.all(tensor == pad.unsqueeze(0).unsqueeze(0), dim=2)
    first_pad_pos = torch.argmax(mask.float(), dim=1)
    has_pattern = torch.any(mask, dim=1)
    return torch.where(has_pattern, first_pad_pos, tensor.shape[1])

def predict(model, src):
    src = src.to(device)

    seq_len = count_before_first_pad(src, pad)
    
    assert (
        len(seq_len) == src.size(0)
    ), "Batch size ("+str(src.size(0))+") doesn't correspond to the batch padder ("+str(len(seq_len))+")"

    output =model.predict_next_sentence(EmbeddingsBatch(src)).seqs
    return output


def criterion(src, output):
    src = src.to(device)
    trg = src[:, 1:].contiguous()

    mask = ~(src == pad).all(dim=-1) #[batch, seq]
    mask = mask[:, 1:].reshape(-1)

    predicted_seqs = output[:, :-1].contiguous()

    # only measure distance over `target_mask = True` positions
    flattened_predictions = predicted_seqs.view(-1, d_emb)[mask]
    flattened_target = trg.view(-1, d_emb)[mask]

    # Cast features to float32 before computing the loss:
    reconstruction_loss, _ = compute_standard_mse(flattened_predictions.float(), flattened_target.float())
    cossim_ = cossim(flattened_predictions.float(), flattened_target.float(), torch.full((1,), 1).to(device))

    return cossim_, reconstruction_loss.mean()

from fairseq2.nn.utils.gradient import clip_gradient_norm

def process_gradients(model, loss_scaler):
    # Normalize gradients
    """
    Normalize and clip the gradients
    """
    # this raw grad norm is only used for debugging
    raw_grad_norm = clip_gradient_norm(model, max_norm=None)

    # undo the GradScaler's scaling before clipping
    loss_scaler.unscale_gradients_()

    # Clip gradients
    # If DDP, we use torch.nn.utils.clip_grad_norm_, if FSDP,
    # we use torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
    # this method handles the fact that gradients might be sharded across ranks.
    grad_norm = clip_gradient_norm(model, max_norm=1000)
    
    return grad_norm, raw_grad_norm

# %%
def validation(epoch, cpt, model, valloader, val_list):
    cossim_loss = 0; mse_loss = 0
    model.eval()
    with torch.no_grad():
        for src in valloader:
            output = predict(model, src)
            cossim_, mse_ = criterion(src, output)
            cossim_loss += cossim_.item(); mse_loss += mse_.item()
        cossim_loss = cossim_loss/len(valloader)
        mse_loss = mse_loss/len(valloader)
        val_list.append([cossim_loss, mse_loss])
        print('Epoch', epoch+1, 'Part', cpt, "Cossim", cossim_loss, "MSE", mse_loss)
        wandb.log({'Cossim': cossim_loss, 'MSE': mse_loss})
    model.train()
    return val_list, cpt+1


def autoregr_infer(model, src):
    seq = 20-1
    list_autoregr = []
    model.eval()
    with torch.no_grad():
        output = predict(model, src)
        autoregr = torch.cat((src, output), dim=1)
        for i in range(seq-1):
            out = predict(model, autoregr)
            output = out[:, -1].unsqueeze(1)
            autoregr = torch.cat((autoregr, output), dim=1)
        list_autoregr.append(autoregr[:, 1:])
    return torch.cat(list_autoregr, dim=0)


def calculate_score(output, targets):
    seq = 20-1

    score_one_sum = 0
    score_sum_pad = torch.zeros(seq).to(device)
    pad_nbr_sum = torch.zeros(seq).to(device)

    for batch, target in enumerate(targets):
        out = output[batch][:len(target)].to(device)
        
        score = cos(out, target)

        score_one_sum += score.mean()
        score_sum_pad += F.pad(score, (0, seq - len(score)))
        pad_nbr_sum += F.pad(torch.ones(len(score)), (0, seq - len(score))).to(device)

    paragraphed_score = score_sum_pad/pad_nbr_sum
    paragraphed_score = [round(elem.item(), 2) for elem in paragraphed_score]
    final_score = score_one_sum.item()/len(targets)
    return final_score, paragraphed_score


def test(model, test_data, config):
    
    sonarprompt, sonaroutput = test_data
    
    output_autoregr = autoregr_infer(model, sonarprompt)
    final_sonar, paragraphed_sonar = calculate_score(output_autoregr, sonaroutput)

    print('Sonar score:', final_sonar, paragraphed_sonar)
    wandb.log({'Final sonar': final_sonar, 'Paragraphed sonar': paragraphed_sonar})

    torch.cuda.empty_cache(); del sonarprompt, sonaroutput, output_autoregr; gc.collect()

    if final_sonar > 0.5:
        date = str(datetime.date.today())[5:]
        torch.save(model.state_dict(), '/workspace/eloise/sentemb/Base_LCM_orig_' + str(round(final_sonar, 2)) +  '_' + str(config.lcm_num_layers) + '-' + str(config.lcm_ffn_inner_dim) + '-' + str(config.model_dim)+ '_' + date +'.pth')

# %%
from lcm.models.base_lcm.archs import base_lcm_max
from lcm.models.base_lcm.builder import create_base_lcm_model

def objective(config, data):
    
    traindata, valloader, test_data = data
    
    model = create_base_lcm_model(base_lcm_max(config), device=device)
    model_save = model

    lr_schedule = config.lr_schedule
    lr = 0.004
    num_lr_warmup_steps: int = 800
    adam_eps: float = 1e-6
    weight_decay = 0.1
    start_lr: float = 1e-7
    final_lr: float = 1e-5
    max_steps: int = 10_000
    turn_off_grad_normalization = False

    optimizer = AdamW(
                model.parameters(),
                lr=lr,
                betas=tuple([0.9, 0.98]),
                eps=adam_eps,
                use_fp32=True,
                weight_decay=weight_decay,
            )
            

    lr_scheduler = build_lr_scheduler(
                optimizer=optimizer,
                schedule=lr_schedule,
                lr=lr,
                warmup_steps=num_lr_warmup_steps,
                start_lr=start_lr,
                final_lr=final_lr,
                max_steps=max_steps,
                stage_ratio=tuple([0.1, 0.4, 0.5]),
            )

    loss_scaler = DynamicLossScaler(optimizer,min_scale=0.0001,enabled= False ,gang = Gang)

    val_list=[]
    cpt = 1
    loss_type=1 #COSSIM:0   MSE:1
    patience = 3

    checker = 6400/config.batch_size
    
    step_nr=1    

    print('')
    
    for epoch in itertools.cycle(range(10)):
        if epoch!=0:
                torch.cuda.empty_cache(); del epoch_traindata, trainloader; gc.collect()
        if epoch%10==0:
            epoch_traindata = traindata
        else:
            epoch_traindata = torch.from_numpy(np.load('/workspace/eloise/sentemb/data/'+str(epoch%10)+'00k.npy'))
            
        trainloader = DataLoader(epoch_traindata, config.batch_size)
        
        model.train()

        for i, src in enumerate(trainloader):
            output = predict(model, src)
            loss = criterion(src, output)[loss_type]

            loss_scaler.backward(loss)
            grad_norm, raw_grad_norm = process_gradients(model, loss_scaler)

            _, scale_result = loss_scaler.run_optimizer_step(step_nr)

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            

            step_nr+=1
            
            if i!=0 and i%checker==0:
                val_list, cpt = validation(epoch, cpt, model, valloader, val_list)
                
                if math.isnan(val_list[-1][1]):
                    print('math is nan', val_list[-1][1])
                    if len(val_list) > 1:
                        test(model_save, test_data, config)
                    return
                
                #to save time, we cut the feed when the score isn’t dropping quickly enough:
                if val_list[0][1]> 0.8: wandb.log({'Cossim': val_list[-1][0], 'MSE': val_list[-1][1]}); test(model_save, test_data, config); return
                if cpt==5 and val_list[-1][1]> 0.1: wandb.log({'Cossim': val_list[-1][0], 'MSE': val_list[-1][1]}); test(model_save, test_data, config); return
                if cpt==30 and val_list[-1][1]> 5e-5: wandb.log({'Cossim': val_list[-1][0], 'MSE': val_list[-1][1]}); test(model_save, test_data, config); return
                
                #early stopping code:
                if len(val_list) >= 2:
                    loss_list = [x[loss_type] for x in val_list]
                    if loss_list[-1] < min(loss_list[:-1]):#it performs better, we save the model
                        model_save = copy.deepcopy(model)
                    elif (len(loss_list) - loss_list.index(min(loss_list))) > patience: #no better model in the last epochs
                        print('Best: Part', cpt-1-patience, 'Cossim', val_list[-1-patience][0], 'MSE', val_list[-1-patience][1])
                        wandb.log({'Cossim': val_list[-1-patience][0], 'MSE': val_list[-1-patience][1]})
                        torch.cuda.empty_cache(); del src, output, model, optimizer, valloader, trainloader, epoch_traindata, traindata, loss; gc.collect()
                        
                        #compute Final score
                        test(model_save, test_data, config)
                        return

# %%
sweep_conf = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Final sonar"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32, 64]},
        #"scheduler": {"values": [True, False]},
        #"T_0": {"min": 10, "max": 20000},
        #"lr": {"distribution": "log_uniform_values", "min": 1e-9, "max": 1e-1},
        #"start_lr": {"distribution": "log_uniform_values", "min": 1e-9, "max": 1e-5},
        #"lr_time": {"min": 10, "max": 10000},
        
        #"warmup_steps": {"values": [100, 500, 1000, 2000, 5000]},
        
        #"weight_decay": {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.1},
        
        "lr_schedule": {"values": ['noop', 'myle', 'cosine', 'wsd', 'polynomial']},
        #"clip_grad": {"values": [True, False]},
        
        "model_dim": {"values": [512, 1024, 2048]},

        #"frontend_dropout_p": {"min": 0.0, "max": 0.5},
        #"frontend_pre_linear_init_fn": {"values": ['xavier', 'sonar', 'zero', 'trunc_normal', 'kaiming_uniform', 'none']},
        #"frontend_scale_embeddings": {"values": [True, False]},
        #"frontend_weight_normalization": {"values": [True, False]},

        #"lcm_final_dropout_p": {"min": 0.0, "max": 0.5},
        #"lcm_attention_dropout_p": {"min": 0.0, "max": 0.5},
        #"lcm_dropout_p": {"min": 0.0, "max": 0.5},
        "lcm_ffn_inner_dim": {"values": [1, 2, 4]},
        "lcm_num_layers": {"values": [2, 8, 15, 24, 32]},
        #"lcm_pos_embedding_style": {"values": ["rope", "sine", "learned", "none"]},
        #"lcm_use_swiglu": {"values": [True, False]},
        #"lcm_ffn_inner_activation_name": {"values": ["relu", "tanh", "elu", "leaky_relu", "prelu", "selu", "gelu", "silu", "softsign", "sigmoid", "hardsigmoid", None]},
        #"lcm_ffn_inner_activation_name": {"values": ["relu", "tanh", "elu", "leaky_relu", "selu", "gelu", "silu", "softsign", "sigmoid", "hardsigmoid", None]},
        #"lcm_layer_normalization_style": {"values": ["standard", "fp32", "rms", "unit"]},
        #"lcm_norm_order_style": {"values": ['pre', 'post', 'normformer']},
        #"lcm_final_norm_order_style": {"values": ['pre', 'post', 'normformer']},
        #"lcm_enable_qk_layernorm": {"values": [True, False]},
        #"lcm_mha_qkv_weight_normalization": {"values": [True, False]},
        #"lcm_mha_output_weight_normalization": {"values": [True, False]},
        #"lcm_mha_output_proj_bias": {"values": [True, False]},
        #"lcm_attention_output_init_fn": {"values": ['xavier', 'sonar', 'zero', 'trunc_normal', 'kaiming_uniform', 'none']},

        #"postnet_dropout_p": {"min": 0.0, "max": 0.5},
        #"postnet_linear_init_fn": {"values": ['xavier', 'sonar', 'zero', 'trunc_normal', 'kaiming_uniform', 'none']},
        #"postnet_weight_normalization": {"values": [True, False]},
        #"postnet_layer_normalization_style": {"values": ["standard", "fp32", "rms", "unit"]},
        #"postnet_activation_name": {"values": ["relu", "tanh", "elu", "leaky_relu", "prelu", "selu", "gelu", "silu", "softsign", "sigmoid", "hardsigmoid", None]},
    },
}

print('Program launched')
np_data = np.load('/workspace/eloise/sentemb/data/100k_1k_0.npz')
print('Data loaded')
traindata = torch.from_numpy(np_data['train'])
print('traindata ready')
valloader = DataLoader(torch.from_numpy(np_data['val']), batch_size=64)
print('valloader ready')
test_data = torch.load('/workspace/eloise/sentemb/data/test_sonarprompt_sonaroutput.pth')
torch.cuda.empty_cache(); del np_data; gc.collect()
print('test_data ready')


wandb.init(settings=wandb.Settings(save_code=False))

def main():
    wandb.init()
    try:
        objective(wandb.config, [traindata, valloader, test_data])
    except Exception as e:
        print(e)
        traceback.print_exc()
    
sweep_id = wandb.sweep(sweep_conf, project="LCM")

wandb.agent(sweep_id, function=main)


