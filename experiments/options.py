import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='LN_prompt')

# --------------------
# DataLoader Options
# --------------------

# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
parser.add_argument('--data_dir', type=str, default='/isize2/sain/data/Sketchy/') 
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# ----------------------
# Training Params
# ----------------------

parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=192)
parser.add_argument('--workers', type=int, default=128)

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# ----------------------
# Auxiliary Losses
# ----------------------

# Set > 0 to add an auxiliary classification loss (CrossEntropy) during training.
parser.add_argument('--cls_loss_weight', type=float, default=0.5)

# Triplet loss weight (main retrieval loss).
parser.add_argument('--triplet_loss_weight', type=float, default=1.0)

# Symmetric contrastive loss (InfoNCE) on sketch/image features.
parser.add_argument('--contrastive_loss_weight', type=float, default=0.5)
parser.add_argument('--contrastive_temperature', type=float, default=0.07)

# Loss balancing / scheduling to prevent one loss from dominating due to scale.
# - fixed: raw weighted sum
# - ema_normalize: divide each loss by its EMA magnitude before summing
parser.add_argument('--loss_balance', type=str, default='ema_normalize', choices=['fixed', 'ema_normalize'])
parser.add_argument('--loss_ema_beta', type=float, default=0.98)
parser.add_argument('--loss_warmup_steps', type=int, default=500)

# Negative mining strategy for the triplet loss.
# - random: use pre-sampled neg_tensor from dataset
# - batch_hard: mine hardest negative from in-batch images with different category
parser.add_argument('--negative_mining', type=str, default='batch_hard', choices=['random', 'batch_hard'])

# Data augmentation
parser.add_argument('--use_augmentation', action='store_true', default=True)
parser.add_argument('--no_augmentation', action='store_false', dest='use_augmentation')

opts = parser.parse_args()
