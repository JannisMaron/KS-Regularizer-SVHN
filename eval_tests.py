import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from sklearn.manifold import TSNE

import cv2

import util

# Get GPU
device = util.get_default_device()

classes = (0, 1, 2, 3, 4,
    5, 6, 7, 8, 9)  

custom_labels = ["0", "0 [Adv]", 
                 "1", "1 [Adv]", 
                 "2", "2 [Adv]", 
                 "3", "3 [Adv]", 
                 "4", "4 [Adv]", 
                 "5", "5 [Adv]", 
                 "6", "6 [Adv]", 
                 "7", "7 [Adv]", 
                 "8", "8 [Adv]", 
                 "9", "9 [Adv]"]


def run_report(run_info):
    """Prints run settings"""
    
    print("  Loss Function:", run_info["loss_type"])
    print()
    print("  Alpha:", run_info["weight_scale"])
    print("  Weights:", run_info["weights"])
    print()
    print("  Using Extra Layer:", run_info["extra_layer"])
    print("  Regularize Latent space:", run_info["latent_reg"])
    print()
    print("  Trained with balanced Dataloader:", run_info["balanced_load"])
    print()


################################################################################
# Report Losses
################################################################################
  
def plot_train_report(report, data_path):
    """Plots Train and Val Losses/Accuracy of the training setting"""
    
    tl = np.array(report[0][0])
    tl_partial = np.array(report[0][1])
    
    vl = np.array(report[1][0]["Adv Loss"])
    vl_partial = np.array(report[1][1]["Adv Partial Loss"])
    
    clean_acc = np.array(report[2]["Clean Acc"])
    adv_acc = np.array(report[2]["Adv Acc"])
    
    
    epochs, num_partial = tl_partial.shape
    
    x = np.arange(epochs)
    
    # Total Loss
    fig,ax1 = plt.subplots(2,figsize=(12,12))
    ax1[0].plot(x, tl, label = "Train Loss")
    ax1[1].plot(x, vl, label = "Val Loss")
    
    for a in ax1:
        a.legend()
        a.grid()
        
    plt.tight_layout()  
    plt.savefig(data_path + "/Total Loss.png")
    plt.show()
    
    
    # Partial Losses
    
    if num_partial > 1:
        fig,ax2 = plt.subplots(num_partial,2,figsize=(12,12))
        
        for i in range(num_partial):
            
            ax2[i,0].plot(x,tl_partial[:,i],label = "Train Partial Loss")
            ax2[i,1].plot(x,vl_partial[:,i],label = "Val Partial Loss")
    
        for a in ax2:
            a[0].legend()
            a[1].legend()
            a[0].grid()
            a[1].grid()
            
        plt.tight_layout()
        plt.savefig(data_path + "/Train Partial Loss.png")
        plt.show()
    
    
    
    # Accuracy
    fig,ax1 = plt.subplots(1,figsize=(12,12))
    ax1.plot(x, clean_acc, label = "Clean Acc")
    ax1.plot(x, adv_acc, label = "Adv Acc")
    
    ax1.legend()
    plt.tight_layout() 
    ax1.grid()
    plt.savefig(data_path + "/Train Accuracy.png")
    plt.show()
    
    pass


def plot_val_report(report, data_path):
    
    vl = report[1][0]
    vl_partial = report[1][1]
    accuracy = report[2]
    
    
    epochs, num_partial = np.array(report[0][1]).shape
    x = np.arange(epochs)
    
    
    fig,ax = plt.subplots(1,figsize=(12,12))
    # Loss
    for key, value in vl.items():
        loss = np.array(value)
        ax.plot(x, loss, label=key)
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.tight_layout()
    plt.savefig(data_path + "/Val Loss.png")
    plt.show()
    
    
    # Partial Loss
    loss_types = ["Clean Loss", "Adv Loss", "KS Loss", "KS-Pair Loss", "Cov Loss"]
    if num_partial > 1:
       fig,ax2 = plt.subplots(num_partial,figsize=(12,12))
       
       for i, (key, values) in enumerate(vl_partial.items()):
           
           data = np.array(values)
           for j in range(num_partial):
               ax2[j].grid(True)
               ax2[j].plot(x, data[:,j], label=key) 
               ax2[j].legend(loc='center left', bbox_to_anchor=(1, 0.5))
               ax2[j].set_title(loss_types[j])
               
              
       plt.tight_layout()
       plt.savefig(data_path + "/Val Partial Loss.png")
       plt.show()
       
       
    # Acc
    for key, value in accuracy.items():
       
       acc = np.array(value)
       plt.plot(x, acc, label=key)
       
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.grid()
    plt.savefig(data_path + "/Val Accuracy.png")
    plt.show()
    
    pass


################################################################################
# Accuracy
################################################################################
        
    
def accuracy(model, data, epsilon, alpha, iterations):
    
    # accuracy
    total = 0
    correct = 0
    adv_correct = 0
    
    model.eval()
    
    for i,batch in enumerate(data):
       
        # Get images and labels
        batch = util.to_device(batch, device)
        x, label = batch
      
        # craft Adversarial Samples
        adv_x = util.adv_samples_PGD(model, x, label, epsilon, alpha, iterations)
       
        with torch.no_grad():
            # model outputs
            pred, z, = model(x)
            adv_pred, adv_z = model(adv_x)
            
            # accuracy
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred,1)
            correct += (pred == label).sum().item()
            
            # adv acc
            adv_pred = F.softmax(adv_pred, dim=1)
            adv_pred = torch.argmax(adv_pred,1)
            adv_correct += (adv_pred == label).sum().item()
            
            total += label.size(0)
    
    accuracy = correct / total
    adv_accuracy = adv_correct / total
    
    print("Accuracy: ", np.round(accuracy,5))
    print("Adv Accuracy: ", np.round(adv_accuracy,5))
    
    
################################################################################
# Draw Samples
################################################################################   
def draw_samples(model, data, epsilon, alpha, iterations, num_samples, image_folder):
    
    model.eval()
    
    batch = util.to_device(next(iter(data)),device)
    x, label = batch
    
    x = x[:num_samples]
    label = label[:num_samples]
    
    # craft Adversarial Samples
    adv_x = util.adv_samples_PGD(model, x, label, epsilon, alpha, iterations)
    
    with torch.no_grad():
        # model outputs
        pred, z, = model(x)
        adv_pred, adv_z = model(adv_x)
    
    
    pred = F.log_softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    adv_pred = F.log_softmax(adv_pred, dim=1)
    adv_pred = torch.argmax(adv_pred, dim=1)
    
    imgs = torch.concatenate((x,adv_x)).cpu()
    imgs = np.transpose(imgs,(0,2,3,1))

    preds =  torch.concatenate((pred,adv_pred)).cpu()
    
    fig, ax = plt.subplots(num_samples,2,  figsize=(6, 6))
    for i, ax in enumerate(ax.flat):
        # Plot the image
        ax.imshow(imgs[i])
        ax.axis('off')

        # Set title with ground truth and predicted labels
        title = f"GT: {classes[label[i%num_samples]]}\nPred: {classes[preds[i]]}"
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(image_folder + "/Example Predictions.png")
    plt.show()
    
    
################################################################################
# Confusion Matrix
################################################################################ 
def get_confusion_matrix(model, data, epsilon, alpha, iterations, image_folder):
    
    
    label_pred = []
    label_adv_pred = []
    label_true = []
    
    model.eval()
    
    for i,batch in enumerate(data):
       
        # Get images and labels
        batch = util.to_device(batch, device)
        x, label = batch
      
        # craft Adversarial Samples
        adv_x = util.adv_samples_PGD(model, x, label, epsilon, alpha, iterations)
        
        with torch.no_grad():
            # model outputs
            pred, z, = model(x)
            adv_pred, adv_z = model(adv_x)
        
        # accuracy
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred,1)
        
        # adv acc
        adv_pred = F.softmax(adv_pred, dim=1)
        adv_pred = torch.argmax(adv_pred,1)
    
        label_true.extend(label.data.cpu().numpy())
        label_pred.extend(pred.data.cpu().numpy())
        label_adv_pred.extend(adv_pred.data.cpu().numpy())
     
        
    # Normal Samples
    cf_matrix = confusion_matrix(label_true, label_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    plt.tight_layout()
    sn.heatmap(df_cm, annot=True)
    plt.savefig(image_folder + "/Heatmap.png")
    plt.show()
    
    
    # Adv Samples
    adv_cf_matrix = confusion_matrix(label_true, label_adv_pred)
    adv_df_cm = pd.DataFrame(adv_cf_matrix / np.sum(adv_cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    plt.tight_layout()
    sn.heatmap(adv_df_cm, annot=True)
    plt.savefig(image_folder + "/Adv Heatmap.png")
    plt.show()
    pass

################################################################################
# t-SNE plots
################################################################################ 

def get_tsne(model, data, epsilon, alpha, iterations, samples, image_folder):
    
    all_latents = []
    all_adv_latents = []
    all_labels = []

    
    model.eval()
    batch_size = data.batch_size
    
    for batch_idx, batch in enumerate(data):
        batch = util.to_device(batch, device)
        x, label = batch
        
      
        # craft Adversarial Samples
        adv_x = util.adv_samples_PGD(model, x, label, epsilon, alpha, iterations)
        
        with torch.no_grad():
            # model outputs
            pred, z, = model(x)
            adv_pred, adv_z = model(adv_x)
        
    
        z = z.cpu().detach().numpy()
        adv_z = adv_z.cpu().detach().numpy()
        label = label.cpu().numpy()
        
        all_latents.extend(z)
        all_adv_latents.extend(adv_z)
        all_labels.extend(label)
        
        if (batch_idx+1) * batch_size >= samples:
            break;
    
    
    
    all_latents = np.array(all_latents)
    all_adv_latents = np.array(all_adv_latents)
    all_labels = np.array(all_labels)
    
    
    
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=samples/10)
    tsne_model = tsne.fit_transform(all_latents)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=all_labels, cmap='viridis', edgecolors='k')
    plt.title('t-SNE Plot of Normal Latent Space')
    cbar = plt.colorbar()
    custom_ticks = list(range(10))
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels(classes)

    plt.tight_layout()
    plt.savefig(image_folder + "/tSNE_Normal.png")
    plt.show()
    plt.clf()
    
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=samples/10)
    tsne_model = tsne.fit_transform(all_adv_latents)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=all_labels, cmap='viridis', edgecolors='k')
    plt.title('t-SNE Plot of Adversarial Latent Space')
    cbar = plt.colorbar()
    custom_ticks = list(range(10))
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels(classes)

    plt.tight_layout()
    plt.savefig(image_folder + "/tSNE_Adv.png")
    plt.show()
    plt.clf()
    
    
    combined_latents = np.vstack((all_latents, all_adv_latents))
    combined_labels = np.hstack((2*all_labels, 2*all_labels+1))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=2*samples/10)
    tsne_model = tsne.fit_transform(combined_latents)
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=combined_labels, cmap='tab20', edgecolors='k')
    plt.title('t-SNE Plot of Combined Latent Space')
    
    cbar = plt.colorbar(scatter, ticks=range(len(set(combined_labels))))
    cbar.set_ticks(range(len(custom_labels)))
    cbar.set_ticklabels(custom_labels)
    
    plt.tight_layout()
    plt.savefig(image_folder + "/tSNE_Combined.png")
    plt.show()
    plt.clf()



    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=combined_labels, cmap='viridis', edgecolors='k')
    plt.title('t-SNE Plot of Combined Latent Space')
    
    cbar = plt.colorbar(scatter, ticks=range(len(set(combined_labels))))
    cbar.set_ticks(range(len(custom_labels)))
    cbar.set_ticklabels(custom_labels)
    
    plt.tight_layout()
    plt.savefig(image_folder + "/tSNE_Combined_Alt.png")
    plt.show()
    plt.clf()

    
    pass


################################################################################
# 2D Marginals
################################################################################ 
    
def get_marginals(model, data, epsilon, alpha, iterations, c1, c2, samples, image_folder):
    
    all_latents = []
    all_adv_latents = []
    all_labels = []
    
    
    model.eval()
    batch_size = data.batch_size

    
    for batch_idx, batch in enumerate(data):
        batch = util.to_device(batch, device)
        x, label = batch
        
      
        # craft Adversarial Samples
        adv_x = util.adv_samples_PGD(model, x, label, epsilon, alpha, iterations)
        
        with torch.no_grad():
            # model outputs
            pred, z, = model(x)
            adv_pred, adv_z = model(adv_x)
    
    
        z = z.cpu().detach().numpy()
        adv_z = adv_z.cpu().detach().numpy()
        label = label.cpu().numpy()
        
        all_latents.extend(z)
        all_adv_latents.extend(adv_z)
        all_labels.extend(label)
        
        if (batch_idx+1) * batch_size >= samples:
            break;
    
    
    all_latents = np.array(all_latents)
    all_adv_latents = np.array(all_adv_latents)
    all_labels = np.array(all_labels)
    
    
    # Normal 
    latents_x = all_latents[:,c1]
    latents_y = all_latents[:,c2]
    
    plt.scatter(latents_x, latents_y, c=all_labels, cmap='viridis', edgecolors='k')
    cbar = plt.colorbar()
    custom_ticks = list(range(10))
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels(classes)
    plt.tight_layout()
    plt.savefig(image_folder + "/marginals_Normal.png")
    plt.show()
    plt.clf()

    
    # Adv
    adv_latents_x = all_adv_latents[:,c1]
    adv_latents_y = all_adv_latents[:,c2]
    
    plt.scatter(adv_latents_x, adv_latents_y, c=all_labels, cmap='viridis', edgecolors='k')
    cbar = plt.colorbar()
    custom_ticks = list(range(10))
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels(classes)
    plt.tight_layout()
    plt.savefig(image_folder + "/marginals_Adv.png")
    plt.show()
    plt.clf()
    
    # Combined
    combined_latents = np.vstack((all_latents, all_adv_latents))
    combined_labels = np.hstack((2*all_labels,2*all_labels+1))
    
    combined_latents_x = combined_latents[:,c1]
    combined_latents_y = combined_latents[:,c2]
    
    scatter = plt.scatter(combined_latents_x, combined_latents_y, c=combined_labels, cmap='tab20', edgecolors='k')
    
    cbar = plt.colorbar(scatter, ticks=range(len(set(combined_labels))))
    cbar.set_ticks(range(len(custom_labels)))
    cbar.set_ticklabels(custom_labels)
    
    plt.tight_layout()
    plt.savefig(image_folder + "/marginals_Combined.png")
    plt.show()   
    plt.clf()


    scatter = plt.scatter(combined_latents_x, combined_latents_y, c=combined_labels, cmap='viridis', edgecolors='k')
    
    cbar = plt.colorbar(scatter, ticks=range(len(set(combined_labels))))
    cbar.set_ticks(range(len(custom_labels)))
    cbar.set_ticklabels(custom_labels)
    
    plt.tight_layout()
    plt.savefig(image_folder + "/marginals_Combined_Alt.png")
    plt.show()   
    plt.clf()


    pass

    
################################################################################
# Marginal Distribution
################################################################################ 
    
def marginal_distribution(model, data, epsilon, alpha, iterations, c, image_folder):
    
    
    all_latents = []
    all_adv_latents = []
    
   
    model.eval()
    batch_size = data.batch_size

   
    for batch_idx, batch in enumerate(data):
        batch = util.to_device(batch, device)
        x, label = batch
        
        # craft Adversarial Samples
        adv_x = util.adv_samples_PGD(model, x, label, epsilon, alpha, iterations)
        
        with torch.no_grad():

            # model outputs
            pred, z, = model(x)
            adv_pred, adv_z = model(adv_x)
            
        all_latents.extend(z.detach().cpu().numpy())
        all_adv_latents.extend(adv_z.detach().cpu().numpy())
    
    

    # Ideal GMM
    n = len(data) * batch_size
    gaussian_mixture = torch.concat((torch.randn(int(n/10*9)), 10+torch.randn(int(n/10*1))))
    gaussian_mixture = gaussian_mixture.numpy()
    
    # Clean Latents
    all_latents = np.array(all_latents)[:,c]
    
    # Adversarial Latents
    all_adv_latents = np.array(all_adv_latents)[:,c]
    
    bins = 100
    range_min = min(np.min(gaussian_mixture),np.min(all_latents), np.min(all_adv_latents))
    range_max = max(np.max(gaussian_mixture),np.max(all_latents), np.max(all_adv_latents))
    
    # PDF
    
    plt.hist(gaussian_mixture, bins=bins, range=(range_min, range_max), alpha=1, label="GMM PDF", histtype='step', linewidth=2)
    plt.hist(all_latents, bins=bins, range=(range_min, range_max), alpha=1, label="Normal Latent PDF", histtype='step', linewidth=2)
    plt.hist(all_adv_latents, bins=bins, range=(range_min, range_max), alpha=1, label="Adv Latent PDF", histtype='step', linewidth=2)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(image_folder + "/Latent PDF.png")
    plt.show()
    plt.clf()
    

    # CDF
  
    x_values = np.linspace(range_min, range_max, 100)
    
    plt.plot(x_values, np.cumsum(np.histogram(gaussian_mixture, bins=bins, 
              range=(range_min, range_max))[0]) / len(gaussian_mixture), label="GMM CDF", 
              linewidth=2, drawstyle='steps-post')
    plt.plot(x_values, np.cumsum(np.histogram(all_latents, bins=bins, 
              range=(range_min, range_max))[0]) / len(all_latents),
                 label="Normal Latent CDF", linewidth=2, drawstyle='steps-post')
    plt.plot(x_values, np.cumsum(np.histogram(all_adv_latents, bins=bins, 
              range=(range_min, range_max))[0]) / len(all_adv_latents),
                 label="Adv Latent CDF", linewidth=2, drawstyle='steps-post')
    
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(image_folder + "/Latent CDF.png")
    plt.show()
    plt.clf()


################################################################################
# Supervised Latent Space
################################################################################   
 
def data_supervision(model, data, epsilon, alpha, iterations, image_folder):
    class_dim_statistics = np.zeros((10,10))
    adv_class_dim_statistics = np.zeros((10, 10))
    
    model.eval()
    
    for i, batch in enumerate(data):
        # Get images and labels
        batch = util.to_device(batch, device)
        x, label = batch
    
        adv_x = util.adv_samples_PGD(model, x, label, epsilon, alpha, iterations)
        
        with torch.no_grad():
            # model outputs
            pred, z, = model(x)
            adv_pred, adv_z = model(adv_x)
        
        max_z_dim = torch.argmax(z,1)
        
        for j in range(len(label)):
            true_class = label[j].item()
            max_dim_value = max_z_dim[j].item()
            class_dim_statistics[true_class][max_dim_value] += 1
            
            
        max_adv_z_dim = torch.argmax(adv_z, 1)
        for j in range(len(label)):
            true_class = label[j].item()
            max_adv_dim_value = max_adv_z_dim[j].item()
            adv_class_dim_statistics[true_class][max_adv_dim_value] += 1
         
         
         
    class_dim_statistics /= class_dim_statistics.sum(axis=1, keepdims=True)
    adv_class_dim_statistics /= adv_class_dim_statistics.sum(axis=1, keepdims=True)
    
    combined_class_dim_statistics = class_dim_statistics + adv_class_dim_statistics
    combined_class_dim_statistics /= 2
    
    class_dim_statistics = np.round(class_dim_statistics,3)
    adv_class_dim_statistics = np.round(adv_class_dim_statistics,3)
    combined_class_dim_statistics = np.round(combined_class_dim_statistics,3)


        
    plt.figure(figsize=(12, 8))
    sn.heatmap(class_dim_statistics, annot=True, fmt='g', cmap='viridis', xticklabels=True, yticklabels=classes)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Class')
    plt.title('Heatmap of Class-Dimension Statistics (Original Examples)')
    
    plt.tight_layout()
    plt.savefig(image_folder + "/Supervision Clean.png")
    plt.show()
    
    # Plot heatmap for adversarial examples
    plt.figure(figsize=(12, 8))
    sn.heatmap(adv_class_dim_statistics, annot=True, fmt='g', cmap='viridis', xticklabels=True, yticklabels=classes)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Class')
    plt.title('Heatmap of Class-Dimension Statistics (Adversarial Examples)')
    
    plt.tight_layout()
    plt.savefig(image_folder + "/Supervision Adv.png")
    plt.show()
    
    # Plot heatmap for adversarial examples
    plt.figure(figsize=(12, 8))
    sn.heatmap(combined_class_dim_statistics, annot=True, fmt='g', cmap='viridis', xticklabels=True, yticklabels=classes)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Class')
    plt.title('Heatmap of Class-Dimension Statistics (Combined Examples)')
    
    plt.tight_layout()
    plt.savefig(image_folder + "/Supervision Combined.png")
    plt.show()
    

################################################################################
# Robustness
################################################################################ 
    
def robustness(model, data, epsilons, alphas, iterations = 10):

        
    model.eval()
    
    for epsilon, alpha in zip(epsilons, alphas):

        total = 0
        adv_correct = 0
                
        
        for i,batch in enumerate(data):
           
            # Get images and labels
            batch = util.to_device(batch, device)
            x, label = batch
          
            # craft Adversarial Samples
            adv_x = util.adv_samples_PGD(model, x, label, epsilon, alpha, iterations)
            
            with torch.no_grad():
                # model outputs
                adv_pred, adv_z = model(adv_x)
                        
            # adv acc
            adv_pred = F.softmax(adv_pred, dim=1)
            adv_pred = torch.argmax(adv_pred,1)
            adv_correct += (adv_pred == label).sum().item()
            
            total += label.size(0)
            
        
        adv_accuracy = adv_correct / total
        
        
        print("  ", np.round(epsilon,3), "/", np.round(alpha,3), ":", np.round(adv_accuracy,3))

