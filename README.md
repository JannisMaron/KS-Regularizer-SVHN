# KS-Regularizer-SVHN
Experiment code of the KS-Regularizer on the SVHN Dataset

Train the model using:
main.py --directory "${directory}" \
        --project_name "${project_name}" \
	    --run_name "${run_name[$SLURM_ARRAY_TASK_ID-1]}" \
	    --loss_type "${loss_type}" \
	    --lr ${lr} \
	    --reg_latent ${reg_latent} \
	    --extra_layer ${extra_layer} \
	    --balanced_load ${balanced_load} \
	    --alpha_clean ${alpha_clean} \
	    --alpha_adv ${alpha_adv} \
	    --alpha_ks ${alpha_ks} \
	    --alpha_ks_pair ${alpha_ks_pair} \
	    --alpha_cov ${alpha_cov}

    Model File save location
        directory : directory

    Weights & Biases tracking
        project_name : project name
        run_name : run name

    Regularizer
        loss_type = {
            clean_loss : Regular Training,
            adv_loss : Adversarial Training,

            base_KS_loss : KS-regularizer without adversarial classification loss,
            adv_KS_loss : KS-regularizer with added adversarial classification loss,
            sup_KS_loss : KS-regularizer with supervised KS Loss,
            inv_sup_KS_loss : KS Regularizer with Inverse KS Loss,
            inv_sup_KS_pair_loss : KS Regularizer with Inverse KS Pair Loss,
        }

    Trainings setup
        lr : max learning rate for CLR Schedular
        reg_latent: {
            True : regularize second to last layer in network,
            False : regularize output of network,
        }
        balanced_load : balanced dataloaders with equal amount of classes per batch

    Hyperparameter weight scaling:
        alpha_clean : Weight for regular classification loss
        alpha_adv : Weight for adversarial classification loss
        alpha_ks : Weight for KS, Supervised KS, and Inverse Supervised KS loss
        alpha_ks_pair : Weight for KS Pair and Inverse KS Pair loss
        alpha_cov : Weight for Covariance loss


Evaluate model with:
eval.py --directory "${directory}"

    directory : path to model file