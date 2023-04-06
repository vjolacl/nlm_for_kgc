import torch
import os

class customdata(torch.utils.data.Dataset):
    
    def __init__(self, token_tensor):
        self.token_tensor = token_tensor
    
    def __len__(self):
        return self.token_tensor.size()[0]
    
    def __getitem__(self, idx):
        item = self.token_tensor[idx]
        return item

def generate_bert_embeddings(data, model):
    model.eval()
    # Generate embeddings
    output = model(data)
    hidden_states = output["hidden_states"]
    last4_layers = hidden_states[-4:]
    
    return last4_layers

def concat_hidden_states(hidden_states):
    # Concat last 4 hidden states for token 0 = CLS token, cls_embdd = hidden_states[:][:][0]
    embdd_concatenated = []

    # For each text sequence...
    for index in range(len(hidden_states[0])):
        # Concatenate the vectors from the last four layers. Each layer vector is 768 values, so    `cat_vec` is length 4*768.
        cat_vec = torch.cat((hidden_states[-1][index][0], hidden_states[-2][index][0],
                             hidden_states[-3][index][0], hidden_states[-4][index][0]), dim=0)
        embdd_concatenated.append(cat_vec)

    # Convert output to 2D tensor
    embdd_tensor = torch.stack(embdd_concatenated, 0)

    return embdd_tensor


def load_ent_embeddings(path, device):
    files = os.listdir(path)
    files = sorted(files)
    rel = files[1]
    ent = files[2:]
    print(files)
    
    ent_embedd_raw = []
    for file in ent:
        batch_embedd = torch.load(path + '/' + file, map_location = torch.device(device))
        batch_embedd.to(device)
        batch_cpu = batch_embedd.to(device)
        ent_embedd_raw.append(batch_cpu)
        del batch_embedd
        torch.cuda.empty_cache()
    
    rel_embedd_raw = torch.load(path + '/' + rel, map_location = torch.device(device))
    
    
    return ent_embedd_raw, rel_embedd_raw 