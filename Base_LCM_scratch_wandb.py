# %%
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gc
import numpy as np
import itertools
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_emb = 1024

pad = torch.from_numpy(np.load('/workspace/eloise/sentemb/data/pad.npy')).to(device)

cos = nn.CosineSimilarity(dim=-1)
cossim = nn.CosineEmbeddingLoss()
mse = nn.MSELoss()

def criterion(output, target):
    
    mask = ~(target == pad).all(dim=-1) #[batch, seq]
    target = target[mask]
    output = output[mask]
    
    cossim_ = cossim(output, target, torch.full((1,), 1).to(device))
    mse_ = mse(output, target)
    return [cossim_, mse_]

def check_for_nans(model):
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            print(f"NaN detected in weights of {name}")
            print("Stopping training")
            wandb.log({'Cossim': 10, 'MSE': 10})
            return True
    return False


from torchtune.modules import RotaryPositionalEmbeddings

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        output = self.linear1(x)
        return F.silu(output) * self.linear2(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, heads, dim, dff, pdrop):
        super().__init__()
        
        assert (dim%heads==0), "dim is not a multiple of heads"
        
        self.heads = heads
        self.scale = (dim // heads) ** 0.5

        self.rotary = RotaryPositionalEmbeddings(dim // heads)

        self.q = nn.Linear(dim, dim) #https://x.com/o_v_shake/status/1890052168520192111
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dff),
            SwiGLU(dff),
            nn.Linear(dff, dim), #not sure of that
        )

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.drop1 = nn.Dropout(pdrop)
        self.drop2 = nn.Dropout(pdrop)
        
    def attention(self, x, pad_mask=None):
        b, t, d = x.size()
        
        x = self.norm1(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        q = q.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        k = k.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, t, self.heads, d // self.heads).transpose(1, 2)

        q = self.rotary(q)
        k = self.rotary(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        pad_mask = pad_mask.unsqueeze(1)
        causal_mask = torch.tril(torch.ones(t, t, dtype=torch.int)).to(device)
        mask = (pad_mask & causal_mask).unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        
        return self.drop1(self.o(out))

    def forward(self, x, pad_mask=None):
        x = x + self.attention(x, pad_mask)
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x

class Model(nn.Module):
    def __init__(self, d_emb, layers, heads, hidden_size, dff, pdrop):
        super().__init__()
        
        self.prenorm = nn.LayerNorm(d_emb)
        self.prelinear = nn.Linear(d_emb, hidden_size)
        self.decoder = nn.ModuleList([DecoderBlock(heads, hidden_size, dff, pdrop) for _ in range(layers)])
        self.postlinear = nn.Linear(hidden_size, d_emb)
        self.postnorm = nn.LayerNorm(d_emb)
        
    def forward(self, x):
        
        pad_mask = (x!=pad).all(dim=-1).int()
        
        x = self.prenorm(x)
        x = self.prelinear(x)
        for layer in self.decoder:
            x = layer(x, pad_mask)
        x = self.postlinear(x)
        x = self.postnorm(x)
        return x



def validation(epoch, cpt, model, valloader, val_list): #6.5 sec
    cossim_loss = 0; mse_loss = 0
    model.eval()
    with torch.no_grad():
        for src in valloader:
            src = src.to(device)
            outputs = model(src[:, :-1])
            [cossim_, mse_] = criterion(outputs, src[:, 1:])
            cossim_loss += cossim_.item(); mse_loss += mse_.item()
        cossim_loss = cossim_loss/len(valloader)
        mse_loss = mse_loss/len(valloader)
        val_list.append([cossim_loss, mse_loss])
        print('Epoch', epoch+1, 'Part', cpt, "Cossim", cossim_loss, "MSE", mse_loss)
        wandb.log({'Cossim': cossim_loss, 'MSE': mse_loss})
    model.train()
    return val_list, cpt+1


def autoregr_infer(model, prompts):
    seq = 20-1
    list_autoregr = []
    model.eval()
    with torch.no_grad():
        prompts = prompts.unsqueeze(0)
        for src in prompts:
            src = src.to(device)
            autoregr = torch.cat((src, model(src)), dim=1)
            for i in range(seq-1):
                outputs = model(autoregr)[:, -1].unsqueeze(1)
                autoregr = torch.cat((autoregr, outputs), dim=1)
            list_autoregr.append(autoregr[:, 1:])
    return torch.cat(list_autoregr, dim=0)


def calculate_score(output, targets):
    seq = 20-1

    score_one_sum = 0
    score_sum_pad = torch.zeros(seq).to(device)
    pad_nbr_sum = torch.zeros(seq).to(device)

    for batch, target in enumerate(targets):
        out = output[batch][:len(target)]
        
        score = cos(out, target)

        score_one_sum += score.mean()
        score_sum_pad += F.pad(score, (0, seq - len(score)))
        pad_nbr_sum += F.pad(torch.ones(len(score)), (0, seq - len(score))).to(device)

    paragraphed_score = score_sum_pad/pad_nbr_sum
    paragraphed_score = [round(elem.item(), 2) for elem in paragraphed_score]
    final_score = score_one_sum.item()/len(targets)
    return final_score, paragraphed_score


def test(model, test_data, layers, heads, hidden_dim):
    
    sonarprompt, sonaroutput = test_data
    
    output_autoregr = autoregr_infer(model, sonarprompt)
    final_sonar, paragraphed_sonar = calculate_score(output_autoregr, sonaroutput)

    print('Sonar score:', final_sonar, paragraphed_sonar)
    wandb.log({'Final sonar': final_sonar, 'Paragraphed sonar': paragraphed_sonar})

    if final_sonar > 0.45:
        torch.save(model.state_dict(), '/workspace/eloise/sentemb/Base_LCM_' + str(round(final_sonar, 2)) + '_' + str(layers) + '-' + str(heads) + '-' + str(hidden_dim)+ '.pth')

# %%
def objective(config, data):
    
    traindata, valloader, test_data = data
    
    
    model = Model(d_emb, config.layers, config.heads, config.hidden_dim, 2048, config.pdrop).to(device)
     
    optimizer = AdamW([
        *model.prenorm.parameters(), *model.prelinear.parameters(),
        *model.decoder.parameters(),
        *model.postlinear.parameters(), *model.postnorm.parameters(),],
        lr=config.lr, weight_decay=config.weight_decay
    )
    
    
    model_save = model

    if config.scheduler:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0)
    else:
        scheduler = None
    
    val_list=[]
    cpt = 1
    loss_type=1 #COSSIM:0   MSE:1
    patience = 3

    validations_per_epoch = 10
    
    print('')
    
    
    for epoch in itertools.count(0):
        if epoch==0:
            trainloader = DataLoader(traindata, batch_size=config.batch_size)
        else:
            trainloader = DataLoader(torch.from_numpy(np.load('/workspace/eloise/sentemb/data/'+str(epoch)+'00k.npy')), batch_size=config.batch_size)
        
        checker = len(trainloader) // validations_per_epoch
        
        model.train()
        for i, src in enumerate(trainloader):
            optimizer.zero_grad()
            src = src.to(device)
            outputs = model(src[:, :-1])
            loss = criterion(outputs, src[:, 1:])[loss_type]
            loss.backward()
            

            if config.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                
            if check_for_nans(model):
                break 
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            if i!=0 and i%checker==0:
                print('Avant Epoch', epoch+1, 'Part', cpt)
                val_list, cpt = validation(epoch, cpt, model, valloader, val_list)
                print('Apres Epoch', epoch+1, 'Part', cpt)
                if val_list[0][1]> 0.8: break; wandb.log({'Cosine Similarity': val_list[-1][0], 'MSE': val_list[-1][1]}); return
                if cpt==5 and val_list[-1][1]> 0.1: break; wandb.log({'Cosine Similarity': val_list[-1][0], 'MSE': val_list[-1][1]}); return
                #early stopping code:
                if len(val_list) >= 2:
                    loss_list = [x[loss_type] for x in val_list]
                    if loss_list[-1] < min(loss_list[:-1]):#it performs better, we save the model
                        model_save = model
                    elif (len(loss_list) - loss_list.index(min(loss_list))) > patience: #no better model in the last epochs
                        print('Best: Part', cpt-1-patience, 'Cosine Similarity', val_list[-1-patience][0], 'MSE', val_list[-1-patience][1])
                        wandb.log({'Cosine Similarity': val_list[-1-patience][0], 'MSE': val_list[-1-patience][1]})
                        torch.cuda.empty_cache(); del src, outputs, model, optimizer, valloader, trainloader, loss; gc.collect()
                        
                        #compute Final score
                        if  val_list[-1-patience][1] < 5e-05:
                            test(model_save, test_data, config.layers, config.heasds, config.hidden_dim)
                        return

    
    torch.cuda.empty_cache(); del model; gc.collect()

# %%
sweep_conf = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "Final sonar"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32, 64]},
        "layers": {"values": [8, 16, 24, 32]},
        "heads": {"values": [8, 16]},
        "hidden_dim": {"values": [512, 1024]},
        "pdrop": {"min": 0.0, "max": 0.5},
        "scheduler": {"values": [True, False]},
        "T_0": {"min": 10, "max": 100000},
        "lr": {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.1},
        "weight_decay": {"distribution": "log_uniform_values", "min": 0.00001, "max": 0.1},
        "clip_grad": {"values": [True, False]},
    },
}

np_data = np.load('../data/100k_1k_0.npz')
traindata = torch.from_numpy(np_data['train'])
valloader = DataLoader(torch.from_numpy(np_data['val']), batch_size=64)
test_data = torch.load('../data/test_sonarprompt_sonaroutput.pth')
torch.cuda.empty_cache(); del np_data; gc.collect()

def main():
    wandb.init()
    objective(wandb.config, [traindata, valloader, test_data])
    
sweep_id = wandb.sweep(sweep_conf, project="fake_LCM")

wandb.agent(sweep_id, function=main)


