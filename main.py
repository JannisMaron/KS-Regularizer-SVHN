import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import os

import argparse

import wandb


import util
import weight_estimation
import models
import train
import regularizer


def get_parser():
    parser = argparse.ArgumentParser(description='Train a model using adversarial training.')
    
    parser.add_argument('--project_name', type=str, default="TempTesting")
    parser.add_argument('--run_name', type=str, default="Default Name")
    parser.add_argument('--directory', type=str, default='Temp')
    
    parser.add_argument('--reg_latent', type=str, default='True')
    parser.add_argument('--extra_layer', type=str, default='True')
    parser.add_argument('--balanced_load', type=str, default='True')
    
    parser.add_argument('--loss_type', type=str, default='inv_sup_KS_loss')
    parser.add_argument('--lr',  type=float, default=2e-3)
    
    parser.add_argument('--alpha_clean', type=float, default=1.0)
    parser.add_argument('--alpha_adv', type=float, default=1.0)
    parser.add_argument('--alpha_ks', type=float, default=1.0)
    parser.add_argument('--alpha_ks_pair', type=float, default=1.0)
    parser.add_argument('--alpha_cov', type=float, default=1.0)

    return parser


def main():
    
    # Weight estimation
    print("Estimate Weights")
    weight_estim_fn = weight_estimation.get_estimate_loss_fn(loss_type)
    ks_weight, ks_pair_weight, cov_weight = weight_estim_fn(batch_size, gmm_centers, gmm_std, coup, num_samples=batch_size)

    alpha = torch.tensor([alpha_clean, alpha_adv, alpha_ks, alpha_ks_pair, alpha_cov])
    lamda = torch.tensor([1.0,1.0, ks_weight, ks_pair_weight, cov_weight])
    weights = alpha * lamda
    
    print("Weight Scaling:", alpha)
    print("Weights:", weights)
    print()
    
    
    loss_fn = regularizer.get_loss_fn(loss_type)
    
    
    # Dataset Transforms
    if simple_transforms:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        # has to be tensor
        data_mean = torch.tensor(mean)
        # has to be tuple
        data_mean_int = []
        for c in range(data_mean.numel()):
            data_mean_int.append(int(255 * data_mean[c]))
        data_mean_int = tuple(data_mean_int)
        data_resolution = 32
        transform = transforms.Compose([
            transforms.RandomCrop(data_resolution, padding=int(data_resolution * 0.125), fill=data_mean_int),
            #transforms.RandomHorizontalFlip(),
            util.SVHNPolicy(fillcolor=data_mean_int),
            transforms.ToTensor(),
        ])
    
    # Dataset
    ds = datasets.SVHN(
       root = data_dir,
       split = "train",                         
       transform = transform, 
       download = True,            
    ) 
    train_ds, val_ds, _ = torch.utils.data.random_split(ds, [train_split, val_split, len(ds)-ignore])
    
    
    
    # Dataloaders
    if balanced_load:
        balanced_sampler = util.BalancedSampler(train_ds, num_classes=10, num_samples_per_class=10)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=balanced_sampler)
        balanced_sampler = util.BalancedSampler(val_ds, num_classes=10, num_samples_per_class=10)
        val_dl = DataLoader(val_ds, batch_size=batch_size, sampler=balanced_sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size, shuffle=True)
        
   
    # Model
    model = models.SVHN_PreAct(latent_dim, extra_layer=extra_layer, reg_latent=reg_latent)
    model = util.to_device(model, device)
    
    if optimizer_type == "Adam":
        # Optimizer
        optim = torch.optim.Adam
        optimizer = optim(model.parameters(), lr=lr)
        
        # Scheduler
        sched = torch.optim.lr_scheduler.MultiStepLR
        milestones = [2 * n_epochs // 5, 3 * n_epochs // 5, 4 * n_epochs // 5]
        lr_factor = 0.1
        batches_per_epoch = len(train_dl)
        scheduler = sched(optimizer, milestones=[milestone*batches_per_epoch for milestone in milestones], gamma=lr_factor)
        
    elif optimizer_type == "SGD":
        optim = torch.optim.SGD
        momentum = 0.9
        weight_decay = 0.0005
        optimizer = optim(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
        
        sched = torch.optim.lr_scheduler.CyclicLR
        max_lr = lr
        base_lr = 1e-7
        scheduler = sched(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=2100, mode='triangular')

    
    if logging:
        
        
        run_info_file = file_path + "/run_info.pth"
        torch.save({
            'loss_type': loss_type,
            'weight_scale': alpha,
            'weights': weights,
            'extra_layer': extra_layer,
            'latent_reg': reg_latent,
            'balanced_load': balanced_load},
            run_info_file)
        
        wandb.login()

        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "loss function": loss_type,
                "lr": lr,
                "weight_scale": alpha,
                "weights": weights,
                "extra layer": extra_layer,
                "latent regularization": reg_latent,
                'balanced_load': balanced_load,
		"simple transforms": simple_transforms,
                "Horizontal Flip": False,
                "Momentum": True,
                "Weight Decay": False,
                "Grad Clip": False
                })
    
    
    # Train
    train.fit(n_epochs, model, optimizer, scheduler, train_dl, val_dl, loss_fn,
          gmm_centers, gmm_std, weights, coup,
          logging, file_path)
    
    
    if logging:
        wandb.finish()
    
    pass


if __name__ == "__main__":
    
    use_parser = True
    logging = True
    simple_transforms = False
    optimizer_type = "SGD"
    #project_name = "SVHN RegEvolution Test"
    
    
    print("Using Parser:", use_parser)
    print("Logging:", logging)
    print("Simple Tranforms:", simple_transforms)
    print("Optimizer:", optimizer_type)
    print()
    
    # parser arguments
    if use_parser:
        
        parser = get_parser()
        args = parser.parse_args()
        
        project_name = args.project_name
        run_name = args.run_name
        directory = args.directory
        
        extra_layer = args.extra_layer.lower() == 'true'
        reg_latent = args.reg_latent.lower() == 'true'
        balanced_load = args.balanced_load.lower() == 'true'
        
        loss_type = args.loss_type
        lr = args.lr
        
        alpha_clean = args.alpha_clean 
        alpha_adv = args.alpha_adv 
        alpha_ks = args.alpha_ks 
        alpha_ks_pair = args.alpha_ks_pair 
        alpha_cov = args.alpha_cov
        
    else:
        
        project_name = "TempTesting"
        run_name = "Testing"
        directory = "Temp"
        
        extra_layer = True
        reg_latent = True
        balanced_load = True
        
        loss_type = "inv_sup_KS_pair_loss"
        lr = 2e-3
        
        alpha_clean = 1.0
        alpha_adv = 1.0
        alpha_ks = 1.0
        alpha_ks_pair = 1.0
        alpha_cov = 1.0
        
        
    # fixed hyperparameters
    n_epochs = 50
    batch_size = 100
    
    train_split = 70000
    val_split = 1000
    ignore = (train_split + val_split)
    
    #optim = torch.optim.Adam
    #sched = torch.optim.lr_scheduler.MultiStepLR
    
    latent_dim = 10
    num_clusters = 10
    coup = 0.95
    gmm_centers, gmm_std = util.set_gmm_centers(latent_dim, num_clusters)
    
    
    # Dataset Location
    data_dir = "./data"
    
    file_path =  'models/' + directory
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    print()
    print("Checkpoints: ", file_path)
    print()
    print("Loss Function: ", loss_type)
    print()
    print("Use Extra Layer: ", extra_layer)
    print("Regularize on Latent Space: ", reg_latent)
    print("Balanced Dataloader: ", balanced_load)
    print()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("CUDA NOT AVAILABLE")
        
    #util.set_rng(-1) 
    
    # Get GPU
    device = util.get_default_device()   
    
    
    
    try:
    
        main()
        
    except KeyboardInterrupt:
        print('\n\nSTOP\n\n')
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
        
