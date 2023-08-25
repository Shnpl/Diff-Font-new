"""
Train a diffusion model on images.
"""
import sys
sys.path.append('.')
import os
import json

from pytorch_lightning import loggers as pl_loggers

import pytorch_lightning as pl

from improved_diffusion.diffusion_model_pl import DiffusionModel

model = DiffusionModel()
model.load_from_checkpoint('lightning_logs/diffusion/version_9/checkpoints/73025.ckpt')
model.freeze()
model.eval()
# 1. load the font that we want to generate

# 2. proceed a generation step
# 3. freeze the model. Instead, we optimize the latent style vector with backpropagation 
# 4. save the latent style vector