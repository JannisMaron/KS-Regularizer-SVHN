import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import datetime
import argparse


import util
import checkpoints
import models
import eval_tests

def evaluate(model, data, report):
    
    print("Start Evaluation:", datetime.datetime.now().strftime("%X"))
    
    epsilon = 8/255
    alpha = 2/255
    iterations = 5
    
    print()
    print("Evaluating on:")
    print("  Epsilon:", round(epsilon,4))
    print("  Alpha:", round(alpha,4))
    print("  Iterations:", iterations)
    print()
    
    print("--Run Info--")
    eval_tests.run_report(run_info)
    print()
    
    print("--Plot Losses/Accuracy")
    eval_tests.plot_train_report(report, image_folder)
    eval_tests.plot_val_report(report, image_folder)
    print()
    
    print("--Accuracy--")
    eval_tests.accuracy(model, data, epsilon, alpha, iterations)
    print()
    
    print("--Draw Samples--")
    eval_tests.draw_samples(model, data, epsilon, alpha, iterations, num_samples=2, image_folder=image_folder)
    print()
    
    print("--Confusion Matrix of Outputs--")
    eval_tests.get_confusion_matrix(model, data, epsilon, alpha, iterations, image_folder=image_folder)
    print()
    
    print("--t-SNE Plots of Latent Space--")
    eval_tests.get_tsne(model, data, epsilon, alpha, iterations, samples=1000, image_folder=image_folder)
    print()
    
    print("--2D Marginals of Latent Space--")
    eval_tests.get_marginals(model, data, epsilon, alpha, iterations, c1=0, c2=1, samples=1000, image_folder=image_folder)
    print()
    
    print("--Marginal Distributions of Latent Space--")
    eval_tests.marginal_distribution(model, data, epsilon, alpha, iterations, c=0, image_folder=image_folder)
    print()
    
    print("--Check Supervision--")
    eval_tests.data_supervision(model, data, epsilon, alpha, iterations, image_folder)
    print()
    
    print("--Robustness--")
    print(" --PGD-5:")
    epsilons = np.array([4, 8, 16]) / 255
    alphas = 2/255 * np.ones_like(epsilons)
    iterations = 5
    eval_tests.robustness(model, data, epsilons, alphas, iterations)
    print()

    print(" --PGD-10:")
    epsilons = np.array([4, 8, 16]) / 255
    alphas = 2/255 * np.ones_like(epsilons)
    iterations = 10
    eval_tests.robustness(model, data, epsilons, alphas, iterations)
    print()


    print(" --PGD-20:")
    epsilons = np.array([4, 8, 16]) / 255
    alphas = 2/255 * np.ones_like(epsilons)
    iterations = 20
    eval_tests.robustness(model, data, epsilons, alphas, iterations)
    print()
    
    
    print(" --PGD-50:")
    epsilons = np.array([4, 8, 16]) / 255
    alphas = 2/255 * np.ones_like(epsilons)
    iterations = 50
    eval_tests.robustness(model, data, epsilons, alphas, iterations)
    print()

    print(" --FGSM:")
    epsilons = np.array([4, 8, 16]) / 255
    alphas = 1.25 * epsilons
    iterations = 1
    eval_tests.robustness(model, data, epsilons, alphas, iterations)
    print()

    print("End Evaluation:", datetime.datetime.now().strftime("%X"))


    
    pass

def main():
    
    # Transforms
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Dataset
    test_ds = datasets.SVHN(
       root = data_dir,
       split = "test",                         
       transform = transform, 
       download = True,            
    ) 
    
    test_ds, _ = torch.utils.data.random_split(test_ds, [26000, 32])
        
    
    # Dataloader
    test_dl = DataLoader(test_ds, batch_size, shuffle=True)
    
    # Model
    model = models.SVHN_PreAct(latent_dim, extra_layer=run_info["extra_layer"], 
                               reg_latent=run_info["latent_reg"])
    model = util.to_device(model, device)
    
    model, tl, vl, acc =\
        checkpoints.load(file_path , "/last.pth", model)
        
    evaluate(model, test_dl, (tl, vl, acc))
    
    
    pass



def get_parser():
    parser = argparse.ArgumentParser(description='Evaluation of Trained model')
    
    parser.add_argument('--directory', type=str, default="Temp")
    return parser

if __name__ == "__main__":
    
    use_parser = True
    
    if use_parser:
        parser = get_parser()
        args = parser.parse_args()
        
        directory = args.directory
        
    else:
        directory = "NoPairTest\\Base\\KS Loss (Extra Layer - Latent)"
    
    batch_size = 100
    
    latent_dim = 10
    num_clusters = 10
    coup = 0.95
    gmm_centers, gmm_std = util.set_gmm_centers(latent_dim, num_clusters)
    
    
    # Dataset Location
    data_dir = "./data"
    
    file_path =  'models/' + directory
    print("File Path:", file_path)
    print()
    
    run_info = checkpoints.info_load(file_path, "/run_info.pth")
    
    image_folder = file_path + "/imgs"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    # Get GPU
    device = util.get_default_device()  
    
    try:
    
        main()
        
    except KeyboardInterrupt:
        print('\n\nSTOP\n\n')
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
