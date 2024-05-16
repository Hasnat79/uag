#!/bin/bash
#sbatch --get-user-env=L                #replicate login env

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=moviechat_textrep_ssbd_job       #Set the job name to "JobExample4"
#SBATCH --time=00:30:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1                #Request 1 node
#SBATCH --ntasks-per-node=1        #Request 8 tasks/cores per node
#SBATCH --mem=32G                     #Request 16GB per node 
#SBATCH --output=oops_stat_Out.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:rtx:1           #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=132705889428            #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=hasnat.md.abdullah@tamu.edu    #Send all emails to email_address 

cd $SCRATCH
myproject
showquota
cd /scratch/user/hasnat.md.abdullah/uag/data/oops_dataset
python oops_stat.py > logs.txt