"""
Train a diffusion model on images.
"""
import sys
sys.path.append('.')

import numpy as np
import torch as th

from PIL import Image
import torchvision.transforms as transforms
import blobfile as bf
import torch
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import json
import math
import random
from improved_diffusion.image_datasets import ImageDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import argparse
import functools
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model,
    create_gaussian_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import pytorch_lightning as pl
from improved_diffusion.train_util import TrainLoop
#from resample import LossAwareSampler, UniformSampler


import torch
import os
import json
# Rewrite the code with Pytorch Lightning
class DiffusionModel(pl.LightningModule):
    def __init__(self, hyper_parameters_path):
        super().__init__()
        with open(hyper_parameters_path,'r') as f:
            kwargs = json.load(f)
        
        
        self.diffusion_params = kwargs["model"]["params"].copy()
        del self.diffusion_params["unet_config"]

        self.train_params = kwargs["model"].copy()
        del self.train_params["params"]

        self.model_params = kwargs["model"]["params"]["unet_config"].copy()
        self.model_params['learn_sigma'] = self.diffusion_params['learn_sigma']
        self.train_data_params = kwargs["data"]["train"].copy()
        self.val_data_params = kwargs["data"]["val"].copy()
        self.test_data_params = kwargs["data"]["test"].copy()

        if self.train_data_params["style_dir"]:
            self.model_params['style_average'] = True
        else:
            self.model_params['style_average'] = False
        self.model = create_model(**self.model_params)
        self.diffusion = create_gaussian_diffusion(**self.diffusion_params)

        self.schedule_sampler = create_named_schedule_sampler(self.train_params["schedule_sampler"], self.diffusion)
        hyper_parameters = dict()
        hyper_parameters.update(self.model_params)
        hyper_parameters.update(self.diffusion_params)
        hyper_parameters.update(self.train_params)
        self.save_hyperparameters(hyper_parameters)

        #
    def forward(self, micro_batch, t,micro_cond):
        if not hasattr(self,'path'):
            self.path = f"lightning_logs/version_{self.logger.version}"
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            micro_batch,
            t,
            model_kwargs=micro_cond
            )
        losses = compute_losses()
        return losses
    def training_step(self, batch, batch_idx):
        img_batch, cond_batch = batch
        img_batch = img_batch.to(self.device)
        cond_batch = {key: value.to(self.device) for key, value in cond_batch.items()}
        

        t, weights = self.schedule_sampler.sample(self.train_data_params['batch_size'], self.device)
        
        losses = self.forward(img_batch, t, cond_batch)
        
        loss = (losses["loss"] * weights).mean()
        mse = (losses["mse"] * weights).mean()
        self.log_dict({'loss':loss.detach(),
                       'mse':mse.detach()})
        return loss
    def validation_step(self, batch, batch_idx):
        if self.local_rank == 0:
            img_batch, cond_batch = batch
            img_batch = img_batch.to(self.device)
            cond_batch = {key: value.to(self.device) for key, value in cond_batch.items()}
            
            model_kwargs = {}
            model_kwargs["content_text"] = cond_batch['content_text'].to(self.device)
            model_kwargs["style_image"] = cond_batch['style_image'].to(self.device)
            if self.model_params["use_stroke"]:
                model_kwargs["stroke"] = cond_batch['stroke'].to(self.device)
    
            sample_fn = (
            self.diffusion.p_sample_loop 
            )
            sample = sample_fn(
                self.model,
                (self.val_data_params['batch_size'] ,
                3, self.model_params["image_size"], 
                self.model_params["image_size"]),
                clip_denoised=True,
                model_kwargs=model_kwargs,
                progress=True
            )
            log_images(sample, img_batch, f"{self.path}/val_{self.global_step}.png")
            self.trainer.save_checkpoint(f"{self.path}/checkpoints/{self.global_step}.ckpt")
    def test_step(self, batch, batch_idx):
        img_batch, cond_batch = batch
        img_batch = img_batch.to(self.device)
        cond_batch = {key: value.to(self.device) for key, value in cond_batch.items()}
        
        model_kwargs = {}
        model_kwargs["content_text"] = cond_batch['content_text'].to(self.device)
        model_kwargs["style_image"] = cond_batch['style_image'].to(self.device)
        if self.model_params["use_stroke"]:
            model_kwargs["stroke"] = cond_batch['stroke'].to(self.device)
    
        #NOTE:TEST ONLY
        # model_kwargs["style_image"] = torch.stack([torch.unsqueeze(torch.load(os.path.join(args_dict["data"]["style_dir"],f"{style_num}.pt")),dim=0)]*args_dict["data"]["batch_size"])    
        # t, weights = self.schedule_sampler.sample(batch_size, self.device)
        ##
        sample_fn = (
        self.diffusion.p_sample_loop 
        )
        sample = sample_fn(
            self.model,
            (self.test_data_params['batch_size'] ,
              3, self.model_params["image_size"], 
              self.model_params["image_size"]),
            clip_denoised=True,
            model_kwargs=model_kwargs,
            progress=True
        )
        log_images(sample, img_batch, f"{self.path}/test_{self.global_step}.png")

        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_params['lr'])
        return optimizer
    def train_dataloader(self):
        data = load_data(
        image_size = self.model_params["image_size"],
        **self.train_data_params    
        )
        return data
    def val_dataloader(self) -> EVAL_DATALOADERS:
        data = load_data(
        image_size = self.model_params["image_size"],
        **self.val_data_params    
        )
        return data
    def test_dataloader(self) -> EVAL_DATALOADERS:
        data = load_data(
        image_size = self.model_params["image_size"],
        **self.test_data_params    
        )
        return data

def log_images(generated_images:torch.Tensor,gt_images:torch.Tensor,path:str):
    """
    input should be float [0,1] instead of uint8, BxCxHxW
    """
    generated_images_list = []
    gt_images_list = []
    batchsize = generated_images.shape[0]
    for i in range(batchsize):
        generated_image = ((generated_images[i] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        generated_image = generated_image.permute(1,2,0).contiguous().cpu().numpy()
        generated_image = Image.fromarray(generated_image).resize((64,64))

        gt_image = ((gt_images[i] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        gt_image = gt_image.permute(1,2,0).contiguous().cpu().numpy()
        gt_image = Image.fromarray(gt_image).resize((64,64))
        
        generated_images_list.append(generated_image)
        gt_images_list.append(gt_image)
    
    width_num = 8
    height_num = math.ceil(batchsize/width_num)*2
    target_shape = (width_num*64,height_num*64)
    background = Image.new("RGB",target_shape,(0,0,0,))
    location = (0,0)
    
    for generated_image in generated_images_list:
        background.paste(generated_image,location)
        location = (location[0]+64,location[1]+0)
        if location[0] >= 64*8:
            location = (0,location[1]+64*2)
    location = (0,64)
    for gt_image in gt_images_list:
        background.paste(gt_image,location)
        location = (location[0]+64,location[1]+0)
        if location[0] >= 64*8:
            location = (0,location[1]+64*2)
    background.save(path)

if __name__ == "__main__":
    trainer = pl.Trainer(accumulate_grad_batches=2,
                         max_steps=-1,
                         precision=16,
                         log_every_n_steps=1,
                         devices=[0,1],
                         val_check_interval=400,
                         limit_val_batches=1,
                         num_sanity_val_steps=0
                         )
    model = DiffusionModel("logs/logs_20230727/train_params.json")
    trainer.fit(model)

    #trainer.test(model,ckpt_path="lightning_logs/version_0/checkpoints/epoch=20800.ckpt")