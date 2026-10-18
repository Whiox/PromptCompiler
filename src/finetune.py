# Run from nanoGPT/: python train.py ../training/finetune_promptcompiler_v03.py
import os

# Upstream joins 'data' with dataset; an absolute path selects our external data.
dataset = os.path.abspath('../datasets/v0.3/')
out_dir = '../experiments/nanogpt_v0.3_seed42'
init_from = 'gpt2'

device = 'cuda'
dtype = 'float16'
compile = False
wandb_log = False

block_size = 256
batch_size = 1
gradient_accumulation_steps = 4

learning_rate = 3e-5
decay_lr = False
max_iters = 200
dropout = 0.0

eval_interval = 200
eval_iters = 10
log_interval = 10
always_save_checkpoint = True
