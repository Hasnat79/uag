#!/bin/bash
#sbatch --get-user-env=L                #replicate login env

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=AL_finetune_videollama1_v2_oops_instruct       #Set the job name to "JobExample4"
#SBATCH --time=12:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1                #Request 1 node
#SBATCH --ntasks-per-node=1        #Request 8 tasks/cores per node
#SBATCH --mem=32G                     #Request 16GB per node 
#SBATCH --output=AL_finetune_videollama1_v2_oops_instruct_Out.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:a100:1          #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=132705889428            #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=hasnat.md.abdullah@tamu.edu    #Send all emails to email_address 


cd /scratch/user/hasnat.md.abdullah/uag/foundation_models/Video-LLaMA

torchrun --nproc_per_node=1 train.py --cfg-path  ./train_configs/audiobranch_stage2_finetune.yaml > logs/audio_branch_finetune_videollama1_v2_oops_instruct.txt
