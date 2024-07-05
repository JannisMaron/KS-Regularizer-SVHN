import datetime
import torch

import wandb

import util
import checkpoints

device = util.get_default_device()


def fit(n_epochs, model, optimizer, scheduler, train_dl, val_dl, loss_fn, 
        gmm_centers, gmm_std, weights, coup,
        logging, file_path):
    
    
    print("Start Training:", datetime.datetime.now().strftime("%X"))
    
    tl = []
    tl_partial = []   
    
    vl = {
        "Adv Loss": [],
        "FGSM Loss": [],
        "Big-Eps Loss": [],
        "More-Iter Loss": []
        }
    
    vl_partial =  {
        "Adv Partial Loss": [],
        "FGSM Partial Loss": [],
        "Big-Eps Partial Loss": [],
        "More-Iter Partial Loss": []
        }
    
    accuracy = {
        "Clean Acc": [],
        "Adv Acc": [],
        "FGSM Acc": [],
        "Big-Eps Acc": [],
        "More-Iter Acc": []
        }
    
    
    for epoch in range(n_epochs):
     
        # Training
        train_loss, train_partial_loss =\
            train(model, optimizer, scheduler, train_dl, loss_fn, gmm_centers, gmm_std, weights, coup)
        
        tl.append(train_loss)
        tl_partial.append(train_partial_loss)
        
        # Validation
        val_loss, val_partial_loss, val_acc =\
            val(model, val_dl, loss_fn, gmm_centers, gmm_std, weights, coup)
            
        vl = util.append_to_dict(vl, val_loss)
        vl_partial = util.append_to_dict(vl_partial, val_partial_loss)
        accuracy = util.append_to_dict(accuracy, val_acc)   
        
        
        
        if epoch % 10 == 0:
            print('\n------------------------------')
            print('Time:', datetime.datetime.now().strftime("%X"))
            print("Currend Epoch: ", epoch)
            
            print()
            print(f"Train Loss: {train_loss:.4} / {train_partial_loss}")
            print(f"Val Loss: {val_loss['Adv Loss']:.4} / {val_partial_loss['Adv Partial Loss']}")
            
            
            print()
            print(f"Clean Acc: {val_acc['Clean Acc']:.4}")
            print(f"Adv Acc: {val_acc['Adv Acc']:.4}")
                        
            print()
            print()
            
            #util.plot_acc_progress(epoch, accuracy)
        
       
        
        # Log to weights and biases
        if logging:
            
            # Save Last Epoch    
            checkpoints.save(model, optimizer, scheduler, epoch, tl, tl_partial, 
                           vl, vl_partial, accuracy,
                           file_path, "/last.pth")  
            
            wandb.log({
                "Train Loss": float(train_loss),
                "Train Clean Loss": float(train_partial_loss[0]),
                "Train Adv Loss": float(train_partial_loss[1]),
                "Train KS Loss": float(train_partial_loss[2]),
                "Train KS-Pair Loss": float(train_partial_loss[3]),
                "Train Cov Loss": float(train_partial_loss[4]),

                "Val Loss": float(val_loss["Adv Loss"]),
                "Val Clean Loss": float(val_partial_loss["Adv Partial Loss"][0]),
                "Val Adv Loss": float(val_partial_loss["Adv Partial Loss"][1]),
                "Val KS Loss": float(val_partial_loss["Adv Partial Loss"][2]),
                "Val KS-Pair Loss": float(val_partial_loss["Adv Partial Loss"][3]),
                "Val Cov Loss": float(val_partial_loss["Adv Partial Loss"][4]),
                
                #"Val-FGSM Loss": float(val_loss["FGSM Loss"]),
                #"Val-FGSM Clean Loss": float(val_partial_loss["FGSM Partial Loss"][0]),
                #"Val-FGSM Adv Loss": float(val_partial_loss["FGSM Partial Loss"][1]),
                #"Val-FGSM KS Loss": float(val_partial_loss["FGSM Partial Loss"][2]),
                #"Val-FGSM KS-Pair Loss": float(val_partial_loss["FGSM Partial Loss"][3]),
                #"Val-FGSM Cov Loss": float(val_partial_loss["FGSM Partial Loss"][4]),
                
                #"Val-Big-Eps Loss": float(val_loss["Big-Eps Loss"]),
                #"Val-Big-Eps Clean Loss": float(val_partial_loss["Big-Eps Partial Loss"][0]),
                #"Val-Big-Eps Adv Loss": float(val_partial_loss["Big-Eps Partial Loss"][1]),
                #"Val-Big-Eps KS Loss": float(val_partial_loss["Big-Eps Partial Loss"][2]),
                #"Val-Big-Eps KS-Pair Loss": float(val_partial_loss["Big-Eps Partial Loss"][3]),
                #"Val-Big-Eps Cov Loss": float(val_partial_loss["Big-Eps Partial Loss"][4]),
                
                #"Val-More-Iter Loss": float(val_loss["More-Iter Loss"]),
                #"Val-More-Iter Clean Loss": float(val_partial_loss["More-Iter Partial Loss"][0]),
                #"Val-More-Iter Adv Loss": float(val_partial_loss["More-Iter Partial Loss"][1]),
                #"Val-More-Iter KS Loss": float(val_partial_loss["More-Iter Partial Loss"][2]),
                #"Val-More-Iter KS-Pair Loss": float(val_partial_loss["More-Iter Partial Loss"][3]),
                #"Val-More-Iter Cov Loss": float(val_partial_loss["More-Iter Partial Loss"][4]),
                
                "Clean Acc": float(val_acc["Clean Acc"]),
                "Adv Acc": float(val_acc["Adv Acc"]),
                "FGSM Acc": float(val_acc["FGSM Acc"]),
                "Big-Eps Acc": float(val_acc["Big-Eps Acc"]),
                "More-Iter Acc": float(val_acc["More-Iter Acc"])
                
                })

    print("End Training:", datetime.datetime.now().strftime("%X"))
        
        
        
def train(model, optimizer, scheduler, train_dl, loss_fn, gmm_centers, gmm_std, weights, coup):
    
    
    batch_loss = 0
    batch_partial_losses = 0
    
    
    
    for i,batch in enumerate(train_dl):
        
        batch = util.to_device(batch, device)
        x, label = batch
        
        # Craft advsearial samples
        model.eval()
        adv_x = util.adv_samples_PGD(model, x, label, 8/255, 2/255, 5)
        
        model.train()
        optimizer.zero_grad()
        
        # Clean images
        logits, z = model(x)
        
        # Adversarial images
        adv_logits, adv_z = model(adv_x)
        
        # Calculate loss w.r.t. regularizer
        loss, partial_losses =\
            loss_fn(logits, adv_logits, z, adv_z, label, weights, gmm_centers, gmm_std, coup)
            
            
            
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        #max_grad_norm = 10
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
        
        optimizer.step()   
        scheduler.step()
        
        
        batch_loss += loss.item()
        batch_partial_losses += partial_losses
        
        
    batch_loss /= len(train_dl)
    batch_partial_losses /= len(train_dl) 
    
    return batch_loss, batch_partial_losses
    

def val(model, val_dl, loss_fn, gmm_centers, gmm_std, weights, coup):
    
    
    # loss
    batch_adv_loss = 0
    batch_fgsm_loss = 0
    batch_big_eps_loss = 0
    batch_more_iter_loss = 0
    
    # partial loss
    batch_adv_partial_loss = 0
    batch_fgsm_partial_loss = 0
    batch_big_eps_partial_loss = 0
    batch_more_iter_partial_loss = 0
    
    # accuracy
    batch_clean_acc = 0
    batch_adv_acc = 0
    batch_fgsm_acc = 0
    batch_big_eps_acc = 0
    batch_more_iter_acc = 0
    
    
    model.eval()
    
    for i,batch in enumerate(val_dl):
        
        batch = util.to_device(batch, device)
        x, label = batch
        
        
        # Clean Outputs
        with torch.no_grad():
            clean_logits, clean_z = model(x)
            
            # Clean accuracy
            clean_acc = util.accuracy(clean_logits, label)
            batch_clean_acc += clean_acc



        
        # Adversarials        
        adv_x = util.adv_samples_PGD(model, x, label, 8/255, 2/255, 5)
        with torch.no_grad():
            adv_logits, adv_z = model(adv_x)
            
            # adv accuracy
            adv_acc = util.accuracy(adv_logits, label)
            batch_adv_acc += adv_acc
            
            # adv loss
            loss, partial_loss =\
                loss_fn(clean_logits, adv_logits, clean_z, adv_z, label, weights, gmm_centers, gmm_std, coup)
        
            batch_adv_loss += loss.item()
            batch_adv_partial_loss += partial_loss
        
        
        
        
        # FGSM adversarials
        adv_x = util.adv_samples_PGD(model, x, label, 8/255, 10/255, 1)
        with torch.no_grad():
            adv_logits, adv_z = model(adv_x)
            
            # adv accuracy
            adv_acc = util.accuracy(adv_logits, label)
            batch_fgsm_acc += adv_acc
            
            # adv loss
            loss, partial_loss =\
                loss_fn(clean_logits, adv_logits, clean_z, adv_z, label, weights, gmm_centers, gmm_std, coup)
    
            batch_fgsm_loss += loss.item()
            batch_fgsm_partial_loss += partial_loss
    
    
    
    
        # PGD big eps   
        adv_x = util.adv_samples_PGD(model, x, label, 12/255, 2/255, 5)
        with torch.no_grad():
            adv_logits, adv_z = model(adv_x)
            
            # adv accuracy
            adv_acc = util.accuracy(adv_logits, label)
            batch_big_eps_acc += adv_acc
            
            # adv loss
            loss, partial_loss =\
                loss_fn(clean_logits, adv_logits, clean_z, adv_z, label, weights, gmm_centers, gmm_std, coup)
        
            batch_big_eps_loss += loss.item()
            batch_big_eps_partial_loss += partial_loss
        



        # PGD more iterations
        adv_x = util.adv_samples_PGD(model, x, label, 8/255, 2/255, 20)
        with torch.no_grad():
            adv_logits, adv_z = model(adv_x)
            
            # adv accuracy
            adv_acc = util.accuracy(adv_logits, label)
            batch_more_iter_acc += adv_acc
            
            # adv loss
            loss, partial_loss =\
                loss_fn(clean_logits, adv_logits, clean_z, adv_z, label, weights, gmm_centers, gmm_std, coup)
        
            batch_more_iter_loss += loss.item()
            batch_more_iter_partial_loss += partial_loss
            
            
    # loss
    batch_adv_loss /= len(val_dl)
    batch_fgsm_loss /= len(val_dl)
    batch_big_eps_loss /= len(val_dl)
    batch_more_iter_loss /= len(val_dl)
    
    # partial loss
    batch_adv_partial_loss /= len(val_dl)
    batch_fgsm_partial_loss /= len(val_dl)
    batch_big_eps_partial_loss /= len(val_dl)
    batch_more_iter_partial_loss /= len(val_dl)  
    
    # accuracy
    batch_clean_acc /= len(val_dl)
    batch_adv_acc /= len(val_dl)
    batch_fgsm_acc /= len(val_dl)
    batch_big_eps_acc /= len(val_dl)
    batch_more_iter_acc /= len(val_dl)
    
    
    # total losses
    total_losses = {
        "Adv Loss": batch_adv_loss,
        "FGSM Loss": batch_fgsm_loss,
        "Big-Eps Loss": batch_big_eps_loss,
        "More-Iter Loss": batch_more_iter_loss
        }
    
    # total partial losses
    total_partial_losses = {
        "Adv Partial Loss": batch_adv_partial_loss,
        "FGSM Partial Loss": batch_fgsm_partial_loss,
        "Big-Eps Partial Loss": batch_big_eps_partial_loss,
        "More-Iter Partial Loss": batch_more_iter_partial_loss
        }
    
    # total accuracy
    total_accuracies = {
        "Clean Acc": batch_clean_acc,
        "Adv Acc": batch_adv_acc,
        "FGSM Acc": batch_fgsm_acc,
        "Big-Eps Acc": batch_big_eps_acc,
        "More-Iter Acc": batch_more_iter_acc
        }
    
    
    return total_losses, total_partial_losses, total_accuracies
