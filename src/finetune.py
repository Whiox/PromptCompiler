# Run from nanoGPT/: python train.py ../src/finetune.py
import os

MODEL_VERSION = "v0.4_2"

# Upstream joins 'data' with dataset; an absolute path selects our external data.
dataset = os.path.abspath(f'../datasets/{MODEL_VERSION}/')
out_dir = f'../experiments/nanogpt_{MODEL_VERSION}_seed42'
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
