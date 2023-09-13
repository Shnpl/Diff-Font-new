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


if __name__ == "__main__":
    # init params
    exp_dir = "lightning_logs/diffusion/diffusion_resnet50_new"
    # real batchsize = batchsize*device_num*grad_accumulate_num = 32*3*2 = 192
    with open(os.path.join(exp_dir,'hyperparams.json'),'r') as f:
        hyperparams = json.load(f)
    # init model and trainer
    trainer = pl.Trainer(accumulate_grad_batches=4,
                        max_steps=-1,
                        precision='16-mixed',
                        log_every_n_steps=1,
                        default_root_dir=exp_dir,
                        devices=hyperparams['misc']['devices'],
                        val_check_interval=800,
                        limit_val_batches=1,
                        num_sanity_val_steps=0
                        )
    model = DiffusionModel(hyperparams,exp_dir)
    trainer.fit(model)
    #trainer.test(model)