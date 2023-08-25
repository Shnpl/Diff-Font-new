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
    device_num = 1
    devices = [1]
    log_dir = "lightning_logs/diffusion"
    name = "diffusion_resnet50"

    # real batchsize = batchsize*device_num*grad_accumulate_num = 32*3*2 = 192
    with open(os.path.join(log_dir,name,'train_params.json'),'r') as f:
        hyper_parameters = json.load(f)
        # hyper_parameters["data"]["train"]["batch_size"] = batchsize
    # init model and trainer
    logger = pl_loggers.CSVLogger(log_dir,name=name)
    trainer = pl.Trainer(accumulate_grad_batches=8,
                        max_steps=-1,
                        precision='16-mixed',
                        log_every_n_steps=1,
                        devices=devices,
                        val_check_interval=800,
                        limit_val_batches=1,
                        num_sanity_val_steps=0,
                        logger=logger
                        )
    model = DiffusionModel(hyper_parameters)
    trainer.test(model,ckpt_path="lightning_logs/diffusion/version_9/checkpoints/73025.ckpt")