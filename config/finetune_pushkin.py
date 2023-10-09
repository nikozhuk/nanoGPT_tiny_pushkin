import time

out_dir = 'out-pushkin'
eval_interval = 5
eval_iters = 40 #40
wandb_log = False # feel free to turn on
wandb_project = 'pushkin'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'pushkin'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# pushkin has 627,550 tokens, so 1 epoch ~= 19.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 40
block_size = 64

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
compile = False