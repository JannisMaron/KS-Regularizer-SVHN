import checkpoints

import os
import matplotlib.pyplot as plt
import numpy as np

import argparse


def plot_all_accuracies(report, image_folder):
    
    accs = report[2]
    
    fig,ax = plt.subplots(5,figsize=(12,12))
    
    for i,acc in enumerate(accs):
        
        # Clean Acc
        clean_acc = acc["Clean Acc"]
        x = np.arange(len(clean_acc))
        
        ax[0].plot(x, clean_acc, label=directories[i])
        
        ax[0].set_title("Clean Acc")
        ax[0].grid(True)
        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Adv Acc
        adv_acc = acc["Adv Acc"]
        x = np.arange(len(adv_acc))
        
        ax[1].plot(x, adv_acc, label=directories[i])
        
        ax[1].set_title("Adv Acc")
        ax[1].grid(True)
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # FGSM Acc
        fgsm_acc = acc["FGSM Acc"]
        x = np.arange(len(fgsm_acc))
        
        ax[2].plot(x, fgsm_acc, label=directories[i])
        
        ax[2].set_title("FGSM Acc")
        ax[2].grid(True)
        ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Larger Eps
        big_eps_acc = acc["Big-Eps Acc"]
        x = np.arange(len(big_eps_acc))
        
        ax[3].plot(x, big_eps_acc, label=directories[i])
        
        ax[3].set_title("Big Eps Acc")
        ax[3].grid(True)
        ax[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # More Iter
        more_iter_acc = acc["More-Iter Acc"]
        x = np.arange(len(more_iter_acc))
        
        ax[4].plot(x, more_iter_acc, label=directories[i])
        
        ax[4].set_title("More Iter Acc")
        ax[4].grid(True)
        ax[4].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    #plt.title("Total Accuracies")
    plt.savefig(image_folder + "/Total Accuracy.png", bbox_inches='tight')
    plt.show()

def plot_all_losses(report, image_folder):
    
    tls = report[0]
    vls = report[1]
    
    
    fig,ax = plt.subplots(2,figsize=(12,12))
    
    for i,tl in enumerate(tls):
        total_train_loss = tl[0]
        x = np.arange(len(total_train_loss))
        ax[0].plot(x, total_train_loss, label=directories[i])
        
        ax[0].set_title("Total Train Loss")
        ax[0].grid(True)
        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    for i,vl in enumerate(vls):
        
        total_val_loss = vl[0]["Adv Loss"]
        x = np.arange(len(total_val_loss))
        ax[1].plot(x, total_val_loss, label=directories[i])
        
        ax[1].set_title("Total Val Loss")
        ax[1].grid(True)
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    #plt.title("Total Loss")
    plt.savefig(image_folder + "/Total Loss.png", bbox_inches='tight')
    plt.show()
    
    
    
    # Partial Losses
    
    fig,ax = plt.subplots(5,2,figsize=(12,12))
    
    
    for i,tl in enumerate(tls):
    
        partial_train_loss = tl[1]
        partial_train_loss = np.array(partial_train_loss)
       
        
        # Clean Loss
        x = np.arange(len(partial_train_loss[:,0]))
        ax[0,0].plot(x, partial_train_loss[:,0], label=directories[i])
        
        ax[0,0].set_title("Partial Train Clean Loss")
        ax[0,0].grid(True)
        #ax[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
        # Adv Loss
        x = np.arange(len(partial_train_loss[:,1]))
        ax[1,0].plot(x, partial_train_loss[:,1], label=directories[i])
        
        ax[1,0].set_title("Partial Train Adv Loss")
        ax[1,0].grid(True)
        #ax[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
        # KS Loss
        x = np.arange(len(partial_train_loss[:,2]))
        ax[2,0].plot(x, partial_train_loss[:,2], label=directories[i])
        
        ax[2,0].set_title("Partial Train KS Loss")
        ax[2,0].grid(True)
        #ax[2,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
        # KS Pair Loss
        x = np.arange(len(partial_train_loss[:,3]))
        ax[3,0].plot(x, partial_train_loss[:,3], label=directories[i])
        
        ax[3,0].set_title("Partial Train KS Pair Loss")
        ax[3,0].grid(True)
        #ax[3,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Cov Loss
        x = np.arange(len(partial_train_loss[:,4]))
        ax[4,0].plot(x, partial_train_loss[:,4], label=directories[i])
        
        ax[4,0].set_title("Partial Train Cov Loss")
        ax[4,0].grid(True)
        #ax[4,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    
    for i,vl in enumerate(vls):
        
        partial_val_loss = vl[1]["Adv Partial Loss"]
        partial_val_loss= np.array(partial_val_loss)
        
        # Clean Loss
        x = np.arange(len(partial_val_loss[:,0]))
        ax[0,1].plot(x, partial_val_loss[:,0], label=directories[i])
        
        ax[0,1].set_title("Partial Val Clean Loss")
        ax[0,1].grid(True)
        ax[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
        # Adv Loss
        x = np.arange(len(partial_val_loss[:,1]))
        ax[1,1].plot(x, partial_val_loss[:,1], label=directories[i])
        
        ax[1,1].set_title("Partial Val Adv Loss")
        ax[1,1].grid(True)
        ax[1,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
        # KS Loss
        x = np.arange(len(partial_val_loss[:,2]))
        ax[2,1].plot(x, partial_val_loss[:,2], label=directories[i])
        
        ax[2,1].set_title("Partial Val KS Loss")
        ax[2,1].grid(True)
        ax[2,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
        # KS Pair Loss
        x = np.arange(len(partial_val_loss[:,3]))
        ax[3,1].plot(x, partial_val_loss[:,3], label=directories[i])
        
        ax[3,1].set_title("Partial Val KS Pair Loss")
        ax[3,1].grid(True)
        ax[3,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Cov Loss
        x = np.arange(len(partial_val_loss[:,4]))
        ax[4,1].plot(x, partial_val_loss[:,4], label=directories[i])
        
        ax[4,1].set_title("Partial Val Cov Loss")
        ax[4,1].grid(True)
        ax[4,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        
    #plt.title("Partial Losses")
    plt.savefig(image_folder + "/Partial Losses.png", bbox_inches='tight')
    plt.show()
    pass


def plot_mean_partial_losses(report, image_folder):
    
    tls = report[0]
    vls = report[1]
    
    # Partial Losses
    fig,ax = plt.subplots(5,2,figsize=(12,12))
    
    # train Losss
    
    clean_train_loss = []
    adv_train_loss = []
    ks_train_loss = []
    ks_pair_train_loss = []
    cov_train_loss = []
    
    # partial losses
    for i,tl in enumerate(tls):
    
        partial_train_loss = tl[1]
        partial_train_loss = np.array(partial_train_loss)
        
        clean_train_loss.append(partial_train_loss[:,0])
        adv_train_loss.append(partial_train_loss[:,1])
        ks_train_loss.append(partial_train_loss[:,2])
        ks_pair_train_loss.append(partial_train_loss[:,3])
        cov_train_loss.append(partial_train_loss[:,4])
    
    
    clean_train_loss = np.array(clean_train_loss)
    adv_train_loss = np.array(adv_train_loss)
    ks_train_loss = np.array(ks_train_loss)
    ks_pair_train_loss = np.array(ks_pair_train_loss)
    cov_train_loss = np.array(cov_train_loss)
    
    # means
    mean_clean_train_loss = clean_train_loss.mean(axis=0)
    mean_adv_train_loss = adv_train_loss.mean(axis=0)
    mean_ks_train_loss = ks_train_loss.mean(axis=0)
    mean_ks_pair_train_loss = ks_pair_train_loss.mean(axis=0)
    mean_cov_train_loss = cov_train_loss.mean(axis=0)
    
    # standard deviation
    std_clean_train_loss = clean_train_loss.std(axis=0)
    std_adv_train_loss = adv_train_loss.std(axis=0)
    std_ks_train_loss = ks_train_loss.std(axis=0)
    std_ks_pair_train_loss = ks_pair_train_loss.std(axis=0)
    std_cov_train_loss = cov_train_loss.std(axis=0)
    
    # plot train loss
    # clean
    x = np.arange(len(mean_clean_train_loss))
    ax[0,0].plot(x, mean_clean_train_loss)
    ax[0,0].fill_between(x, mean_clean_train_loss - std_clean_train_loss, mean_clean_train_loss + std_clean_train_loss, alpha=0.33, label='±1 Std Dev')

    ax[0,0].set_title("Partial Standard Classification Train Loss")
    ax[0,0].grid(True)
    
    # adv
    x = np.arange(len(mean_adv_train_loss))
    ax[1,0].plot(x, mean_adv_train_loss)
    ax[1,0].fill_between(x, mean_adv_train_loss - std_adv_train_loss, mean_adv_train_loss + std_adv_train_loss, alpha=0.33, label='±1 Std Dev')

    ax[1,0].set_title("Partial Adv. Classification Train Loss")
    ax[1,0].grid(True)
    
    # ks
    x = np.arange(len(mean_ks_train_loss))
    ax[2,0].plot(x, mean_ks_train_loss)
    ax[2,0].fill_between(x, mean_ks_train_loss - std_ks_train_loss, mean_ks_train_loss + std_ks_train_loss, alpha=0.33, label='±1 Std Dev')

    ax[2,0].set_title("Partial KS Train Loss")
    ax[2,0].grid(True)
    
    # ks pair
    x = np.arange(len(mean_ks_pair_train_loss))
    ax[3,0].plot(x, mean_ks_pair_train_loss)
    ax[3,0].fill_between(x, mean_ks_pair_train_loss - std_ks_pair_train_loss, mean_ks_pair_train_loss + std_ks_pair_train_loss, alpha=0.33, label='±1 Std Dev')

    ax[3,0].set_title("Partial KS Pair Train Loss")
    ax[3,0].grid(True)
    
    # cov 
    x = np.arange(len(mean_cov_train_loss))
    ax[4,0].plot(x, mean_cov_train_loss)
    ax[4,0].fill_between(x, mean_cov_train_loss - std_cov_train_loss, mean_cov_train_loss + std_cov_train_loss, alpha=0.33, label='±1 Std Dev')

    ax[4,0].set_title("Partial Cov Train Loss")
    ax[4,0].grid(True)
    
    
    
    # val losses
    
    clean_val_loss = []
    adv_val_loss = []
    ks_val_loss = []
    ks_pair_val_loss = []
    cov_val_loss = []
    
    # partial losses
    for i,vl in enumerate(vls):
    
        partial_val_loss = vl[1]["Adv Partial Loss"]
        partial_val_loss= np.array(partial_val_loss)
        
        clean_val_loss.append(partial_val_loss[:,0])
        adv_val_loss.append(partial_val_loss[:,1])
        ks_val_loss.append(partial_val_loss[:,2])
        ks_pair_val_loss.append(partial_val_loss[:,3])
        cov_val_loss.append(partial_val_loss[:,4])
    
    
    clean_val_loss = np.array(clean_val_loss)
    adv_val_loss = np.array(adv_val_loss)
    ks_val_loss = np.array(ks_val_loss)
    ks_pair_val_loss = np.array(ks_pair_val_loss)
    cov_val_loss = np.array(cov_val_loss)
    
    # means
    mean_clean_val_loss = clean_val_loss.mean(axis=0)
    mean_adv_val_loss = adv_val_loss.mean(axis=0)
    mean_ks_val_loss = ks_val_loss.mean(axis=0)
    mean_ks_pair_val_loss = ks_pair_val_loss.mean(axis=0)
    mean_cov_val_loss = cov_val_loss.mean(axis=0)
    
    # standard deviation
    std_clean_val_loss = clean_val_loss.std(axis=0)
    std_adv_val_loss = adv_val_loss.std(axis=0)
    std_ks_val_loss = ks_val_loss.std(axis=0)
    std_ks_pair_val_loss = ks_pair_val_loss.std(axis=0)
    std_cov_val_loss = cov_val_loss.std(axis=0)
    
    # plot val loss
    # clean
    x = np.arange(len(mean_clean_val_loss))
    ax[0,1].plot(x, mean_clean_val_loss)
    ax[0,1].fill_between(x, mean_clean_val_loss - std_clean_val_loss, mean_clean_val_loss + std_clean_val_loss, alpha=0.33, label='±1 Std Dev')

    ax[0,1].set_title("Partial Standard Classification Validation Loss")
    ax[0,1].grid(True)
    
    # adv
    x = np.arange(len(mean_adv_val_loss))
    ax[1,1].plot(x, mean_adv_val_loss)
    ax[1,1].fill_between(x, mean_adv_val_loss - std_adv_val_loss, mean_adv_val_loss + std_adv_val_loss, alpha=0.33, label='±1 Std Dev')

    ax[1,1].set_title("Partial Adv. Classification Validation Loss")
    ax[1,1].grid(True)
    
    # ks
    x = np.arange(len(mean_ks_val_loss))
    ax[2,1].plot(x, mean_ks_val_loss)
    ax[2,1].fill_between(x, mean_ks_val_loss - std_ks_val_loss, mean_ks_val_loss + std_ks_val_loss, alpha=0.33, label='±1 Std Dev')

    ax[2,1].set_title("Partial KS Validation Loss")
    ax[2,1].grid(True)
    
    # ks pair
    x = np.arange(len(mean_ks_pair_val_loss))
    ax[3,1].plot(x, mean_ks_pair_val_loss)
    ax[3,1].fill_between(x, mean_ks_pair_val_loss - std_ks_pair_val_loss, mean_ks_pair_val_loss + std_ks_pair_val_loss, alpha=0.33, label='±1 Std Dev')

    ax[3,1].set_title("Partial KS Pair Validation Loss")
    ax[3,1].grid(True)
    
    # cov 
    x = np.arange(len(mean_cov_val_loss))
    ax[4,1].plot(x, mean_cov_val_loss)
    ax[4,1].fill_between(x, mean_cov_val_loss - std_cov_val_loss, mean_cov_val_loss + std_cov_val_loss, alpha=0.33, label='±1 Std Dev')

    ax[4,1].set_title("Partial Cov Validation Loss")
    ax[4,1].grid(True)
    
    fig.tight_layout()
    plt.savefig(image_folder + "/Average Partial Losses.png", bbox_inches='tight')
    plt.savefig(image_folder + "/Average Partial Losses.pdf", bbox_inches='tight')
    plt.show()
    pass

def mass_evaluate(report):
    
    #plot_all_accuracies(report, image_folder)
    #plot_all_losses(report, image_folder)
    plot_mean_partial_losses(report, image_folder)
    
    pass

def main():
    
    tls, vls, accs = [], [], []
    
    for directory in directories:
        
        file_path = os.path.join(data_folder, directory)
        
        tl, vl, acc = checkpoints.loss_load(file_path , "/last.pth")
        tls.append(tl)
        vls.append(vl)
        accs.append(acc)
    
       
    
    mass_evaluate((tls,vls,accs))
    
def get_parser():
    parser = argparse.ArgumentParser(description='Evaluation of Trained model')
    
    parser.add_argument('--data_folder', type=str, default="Temp")
    parser.add_argument('--mass_folder', type=str, default="MassTemp")
    parser.add_argument('--directories', nargs='+', required=True, help='List of directories to evaluate')
   
    return parser

if __name__ == "__main__":
    
    use_parser = False
    
    if use_parser:
        parser = get_parser()
        args = parser.parse_args()
        
        data_folder = args.data_folder
        mass_folder = args.mass_folder
        directories = args.directories
    else:
    
        data_folder = "models/MT-Final/Layer/KS Loss (No Extra Layer - Logits)"
        mass_folder = "MassEval"
        directories = ["1/KS Loss (No Extra Layer - Logits)",
                       "2/KS Loss (No Extra Layer - Logits)",
                       "3/KS Loss (No Extra Layer - Logits)",
                       "4/KS Loss (No Extra Layer - Logits)",
                       "5/KS Loss (No Extra Layer - Logits)",]
    
    
    image_folder = os.path.join(data_folder, mass_folder, "imgs")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        
    main()