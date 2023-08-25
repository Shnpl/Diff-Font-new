import sys
sys.path.append('.')
sys.path.append('..')
from sklearn.manifold import TSNE
import numpy as np
import os
import random
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from improved_diffusion.font_classifier import FontClassifier_LitModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Train
data_type='800'
model_type = 'resnet50'

log_dir = f'lightning_logs/font_classifier'
logger = CSVLogger(save_dir=log_dir, name=model_type+'_'+data_type)
model = FontClassifier_LitModel(model_type=model_type,data_type=data_type)
#early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=True, strict=True)
saving = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
trainer = pl.Trainer(callbacks=[saving],
                     fast_dev_run=False,
                     logger=logger,
                     max_epochs=32,
                     devices=[0,1,2,3])
trainer.fit(model,ckpt_path='lightning_logs/font_classifier/resnet50_800/version_2/checkpoints/epoch=7-step=49904.ckpt')