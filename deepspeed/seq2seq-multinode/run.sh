#!/bin/bash
#SBATCH --account=hahahaha
#SBATCH --cpus-per-task=10
#SBATCH --nodes=64
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=3:00:00
#SBATCH -o /scratch/aidream/multi-node-seq2seq.out

# Load module
module load python/3.8
module load scipy-stack
module load StdEnv/2020 gcc/9.3.0 cuda/11.4
module load arrow/5.0.0
source ../venv/bin/activate

echo "Module loaded"

# Write hostfile
# Clean previous hostfile
rm -rf hostfile
touch hostfile
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

# Get head node
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

for node in ${nodes}
do
  node_ip=$(srun --nodes=1 --ntasks=1 -w "${node}" hostname --ip-address)
  echo "${node_ip} slots=2" >> hostfile
done

# Run DeepSpeed, we should use the latest version of DeepSpeed
deepspeed --hostfile ./hostfile --num_gpus 2 --num_nodes 64 --master_addr ${head_node_ip} --master_port 25092 train.py --checkpoint_dir /scratch/aidream/trained --dataset_chache_dir /scratch/aidream/datasets --model_name_or_path /scratch/aidream/t5-11b --train_file /scratch/aidream/train_enc_dec.json --validation_file /scratch/aidream/val_enc_dec.json --batch_size 1

# sbatch --account=hahahaha run.sh
