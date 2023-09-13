"""
Train a diffusion model on images.
"""
import sys
sys.path.append('.')
import os
import json
import math
from PIL import Image

import numpy as np

import torch
import torchvision.transforms as transforms
from pytorch_lightning import loggers as pl_loggers

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import tqdm
from improved_diffusion.utils import resize_image
import functools
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    create_model,
    create_gaussian_diffusion,
)
from improved_diffusion.font_classifier import FontClassifier_resnet18,FontClassifier_resnet50,create_style_encoder
#from resample import LossAwareSampler, UniformSampler

# Rewrite the code with Pytorch Lightning
class DiffusionModel(pl.LightningModule):
    def _prepare_hyperparams(self,hyper_parameters):
        if type(hyper_parameters) == dict:
            kwargs = hyper_parameters
        elif type(hyper_parameters) == str:
            with open(hyper_parameters,'r') as f:
                kwargs = json.load(f)
        else:
            raise TypeError("hyper_parameters should be a dict or a str")
        # Misc params(lr,devices,etc.)
        misc = kwargs["misc"].copy()

        # Diffusion params
        diffusion_params = kwargs["diffusion"].copy()

        # Model params(UNET)
        model_params = kwargs["model_params"].copy()
        model_params['learn_sigma'] = diffusion_params['learn_sigma']
        # schedule_sampler_params
        schedule_sampler_params = kwargs["schedule_sampler"]
        # Style Encoder
        style_encoder_params = kwargs["style_encoder"].copy()
        
        # Content Encoder
        if model_params["use_content_encoder"]:
            raise NotImplementedError("Content Encoder is not implemented yet")
        else:
            content_encoder_params = None
        
        # Train data params
        train_data_params = kwargs["data"]["train"].copy()
        train_data_params["image_size"] = model_params["image_size"]

        # Val data params
        val_data_params = kwargs["data"]["val"].copy()
        val_data_params["image_size"] = model_params["image_size"]
        
        # Test data params
        test_data_params = kwargs["data"]["test"].copy()
        test_data_params["image_size"] = model_params["image_size"]

        return misc,diffusion_params, model_params,schedule_sampler_params,style_encoder_params,content_encoder_params, train_data_params, val_data_params, test_data_params
    def _generate_style_vector(self,data_params):
        style_dir = data_params["data_dir"].split('/')[-1]+"_style"
        style_dir = os.path.join(self.path,style_dir)
        data_params["style_dir"] = style_dir
        if not os.path.exists(style_dir):
            os.mkdir(style_dir)
            train_src_dir = data_params["data_dir"]
            styles = os.listdir(train_src_dir)
            self.style_encoder = self.style_encoder.to(torch.device("cuda:0"))
            style_num = len(styles)
            with tqdm.tqdm(total=style_num) as pbar:
                for style in styles:
                    style_path = os.path.join(train_src_dir,style)
                    available_characters_with_ext_raw = os.listdir(style_path)
                    ##
                    available_characters_with_ext = []
                    with open (data_params["char_set"],'r') as f:
                        seen_char = json.load(f)
                    for char_withext in available_characters_with_ext_raw:
                        char = os.path.splitext(char_withext)[0]
                        if char in seen_char:
                            available_characters_with_ext.append(char_withext)
                    
                    ##
                    misc = []
                    for image in available_characters_with_ext:
                        misc.append(Image.open(os.path.join(style_path, image)))
                    sty_emb_tmp = []
                    for image in misc:
                        image = resize_image(image,128).unsqueeze(0).to("cuda:0")
                        #sty_emb_tmp.extend(style_encoder(image)[1].detach())
                        sty_emb_tmp.extend(self.style_encoder(image).detach())
                    sty_emb_tmp = torch.stack(sty_emb_tmp).squeeze()
                    mu = torch.mean(sty_emb_tmp,dim=0)
                    std = torch.std(sty_emb_tmp,dim = 0)
                    new = []
                    for pt in sty_emb_tmp:
                        if torch.norm(pt-mu) < torch.norm(std)*1:
                            new.append(pt)
                    print(len(new))
                    new = torch.stack(new).squeeze()
                    style_emb = torch.mean(new,dim=0)
                    torch.save(style_emb,os.path.join(style_dir,f"{style}.pt"))
                    pbar.update(1)
                #encodings = torch.stack(encodings).cpu().squeeze()
    
    def __init__(self, hyper_parameters,path):
        super().__init__()
        self.path = path
        self.train_params,diffusion_params, model_params,schedule_sampler_params,style_encoder_params,content_encoder_params, self.train_data_params, self.val_data_params, self.test_data_params = self._prepare_hyperparams(hyper_parameters)

        self.diffusion = create_gaussian_diffusion(**diffusion_params)
        self.style_encoder = create_style_encoder(**style_encoder_params)
        self.schedule_sampler = create_named_schedule_sampler(schedule_sampler_params, self.diffusion)         
        self.model = create_model(**model_params)

        self._generate_style_vector(self.train_data_params)
        self._generate_style_vector(self.val_data_params)
        self._generate_style_vector(self.test_data_params)

    def forward(self, micro_batch, t,micro_cond):

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
            if not hasattr(self,'path'):
                self.path = self.logger.log_dir
            if not hasattr(self,'eval_diffusion'):
                self.eval_diffusion_params = self.diffusion_params.copy()
                self.eval_diffusion_params["timestep_respacing"] = "128"
                self.eval_diffusion_params["rescale_timesteps"] = True
                self.eval_diffusion = create_gaussian_diffusion(**self.eval_diffusion_params)

            self.trainer.save_checkpoint(f"{self.path}/checkpoints/{self.global_step}.ckpt")
            #if self.global_step % (self.trainer.val_check_interval*10) == 0:
            img_batch, cond_batch = batch
            img_batch = img_batch.to(self.device)
            cond_batch = {key: value.to(self.device) for key, value in cond_batch.items()}
            
            model_kwargs = {}
            model_kwargs["content_text"] = cond_batch['content_text'].to(self.device)
            model_kwargs["style_image"] = cond_batch['style_image'].to(self.device)
            if self.model_params["use_stroke"]:
                model_kwargs["stroke"] = cond_batch['stroke'].to(self.device)
    
            sample_fn = (
            self.eval_diffusion.p_sample_loop 
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
        data = load_data(**self.train_data_params)
        return data
    def val_dataloader(self) -> EVAL_DATALOADERS:
        data = load_data(**self.val_data_params)
        return data
    def test_dataloader(self) -> EVAL_DATALOADERS:
        data = load_data(**self.test_data_params)
        return data

def log_images(generated_images:torch.Tensor,gt_images:torch.Tensor,path:str):
    """
    input should be float [0,1] instead of uint8, BxCxHxW
    """
    generated_images_list = []
    gt_images_list = []
    batchsize = generated_images.shape[0]
    for i in range(batchsize):
        generated_image = ((generated_images[i] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        generated_image = generated_image.permute(1,2,0).contiguous().cpu().numpy()
        generated_image = Image.fromarray(generated_image).resize((64,64))

        gt_image = ((gt_images[i] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
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