import torch
import numpy as np

import util
import regularizer

def get_estimate_loss_fn(loss_fn):
    
    if loss_fn == "base_KS_loss":
        return estimate_KS_coefficients
    elif loss_fn == "adv_KS_loss":
        return estimate_KS_coefficients
    elif loss_fn == "sup_KS_loss":
        return estimate_sup_KS_coefficients
    elif loss_fn == "inv_sup_KS_loss":
        return estimate_inv_sup_KS_coefficients
    elif loss_fn == "inv_sup_KS_pair_loss":
        return estimate_inv_sup_KS_pair_coefficients
    elif loss_fn == "inv_sup_KS_pair_altered_loss":
        return estimate_inv_sup_KS_pair_altered_coefficients
    
    elif loss_fn == 'clean_loss':
        return no_weights
    elif loss_fn == 'adv_loss':
        return no_weights
    pass

def no_weights(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    return 0.0, 0.0, 0.0

def estimate_KS_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Clean and Adv"""
    
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses, ks_pairlosses = [], [], []

    # Estimate wieghts with gmm samples:
    for i in range(num_samples):
        z, comp = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        z1, comp1 = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        
        #rand_targets = torch.randint(0,10,size=(batch_size,))
        
        
        ksloss = regularizer.ks_loss(z, z1, torch.tensor(comp), gmm_centers=gmm_centers,gmm_std=gmm_std)
        ksloss = ksloss.cpu().detach().numpy()
        
        ks_pairloss = regularizer.ks_pair_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = regularizer.covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ksloss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight


def estimate_sup_KS_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Supervised KS Loss"""
    
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses, ks_pairlosses = [], [], []

    # Estimate wieghts with gmm samples:
    for i in range(num_samples):
        z, comp = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        z1, _ = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        
        #rand_targets = torch.randint(0,10,size=(batch_size,))
        
        
        ksloss = regularizer.sup_ks_loss(z, z1, torch.tensor(comp) ,gmm_centers=gmm_centers,gmm_std=gmm_std)
        ksloss = ksloss.cpu().detach().numpy()
        
        ks_pairloss = regularizer.ks_pair_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = regularizer.covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ksloss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight


def estimate_inv_sup_KS_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Inverse Supervised KS Loss"""
    
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses, ks_pairlosses = [], [], []

    # Estimate wieghts with gmm samples:
    for i in range(num_samples):
        z, comp = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        z1, _ = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        
 #       rand_targets = torch.randint(0,10,size=(batch_size,))
        
        
        ksloss = regularizer.inv_sup_ks_loss(z, z1, torch.tensor(comp), gmm_centers=gmm_centers,gmm_std=gmm_std)
        ksloss = ksloss.cpu().detach().numpy()
        
        ks_pairloss = regularizer.ks_pair_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = regularizer.covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ksloss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight


def estimate_inv_sup_KS_pair_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Inverse Supervised KS Pair Loss"""
    
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses, ks_pairlosses = [], [], []

    # Estimate wieghts with gmm samples:
    for i in range(num_samples):
        z, comp = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        z1, _ = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        
        #rand_targets = torch.randint(0,10,size=(batch_size,))
        
        
        ksloss = regularizer.inv_sup_ks_loss(z, z1, torch.tensor(comp), gmm_centers=gmm_centers,gmm_std=gmm_std)
        ksloss = ksloss.cpu().detach().numpy()
        
        ks_pairloss = regularizer.inv_ks_pair_loss(z, z1)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = regularizer.covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ksloss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight

def estimate_inv_sup_KS_pair_altered_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Inverse Supervised KS Pair Loss"""
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses, ks_pairlosses = [], [], []

    # Estimate wieghts with gmm samples:
    for i in range(num_samples):
        z, comp = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        z1, _ = util.draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        
        #rand_targets = torch.randint(0,10,size=(batch_size,))
        
        
        ksloss = regularizer.inv_sup_ks_loss(z, z1, torch.tensor(comp), gmm_centers=gmm_centers,gmm_std=gmm_std)
        ksloss = ksloss.cpu().detach().numpy()
        
        ks_pairloss = regularizer.inv_ks_pair_altered_loss(z, z1)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = regularizer.covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ksloss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight
    