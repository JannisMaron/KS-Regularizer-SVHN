import torch
import util


def save(model, optimizer, scheduler, epoch, tl, tl_partial,  vl, vl_partial, accuracy,
         file_path, file_name):
    
    path = file_path + file_name
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'schedular_state_dict': scheduler.state_dict(),
    'epoch' : epoch,
    'tl': tl, 
    'tl_partial': tl_partial, 
    'vl': vl, 
    'vl_partial': vl_partial, 
    'acc': accuracy, 
    },path)
    
    
    
    
def load(file_path, file_name, model):
    
    path = file_path + file_name
    
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    tl = checkpoint['tl']
    tl_partial = checkpoint['tl_partial']

    vl = checkpoint['vl']
    vl_partial = checkpoint['vl_partial']
    acc = checkpoint['acc']
    
    
    model.eval()
    
    return model, (tl, tl_partial), (vl, vl_partial), acc


def info_load(file_path, file_name):
    
    path = file_path + file_name
    checkpoint = torch.load(path)
    
    info ={
        "loss_type": checkpoint["loss_type"],
        "weight_scale": checkpoint["weight_scale"],
        "weights": checkpoint["weights"],
        "extra_layer": checkpoint["extra_layer"],
        "latent_reg": checkpoint["latent_reg"],
        "balanced_load": checkpoint["balanced_load"],
        }
    return info

def loss_load(file_path, file_name):
    
    path = file_path + file_name
    checkpoint = torch.load(path)
    
    tl = checkpoint['tl']
    tl_partial = checkpoint['tl_partial']

    vl = checkpoint['vl']
    vl_partial = checkpoint['vl_partial']
    acc = checkpoint['acc']
    
    
    
    return (tl, tl_partial), (vl, vl_partial), acc
    
