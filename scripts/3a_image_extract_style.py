import argparse
import os
import sys
sys.path.append('.')
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import torch 
import torch.distributed as dist

from PIL import Image

import json
import math
import random
import torch
#from improved_diffusion.unet import style_encoder_textedit_addskip
from improved_diffusion.font_classifier import FontClassifier_resnet50 as FontClassifier
import tqdm


if __name__ == "__main__":
    font_dir = "datasets/CFG/font_extra_800_val"
    styles = os.listdir(font_dir)
    #style_encoder = style_encoder_textedit_addskip()
    style_encoder = FontClassifier(use_fc=False)
    style_encoder.eval()
    style_encoder_state_dict = torch.load("lightning_logs/font_classifier/resnet50_800/version_0/checkpoints/epoch=12-step=137227.ckpt",map_location='cpu')
    for k in list(style_encoder_state_dict['state_dict'].keys()):
        if k.startswith('model.'):
            style_encoder_state_dict['state_dict'][k[len("model."):]] = style_encoder_state_dict['state_dict'][k]
            del style_encoder_state_dict['state_dict'][k]
    style_encoder.load_state_dict(style_encoder_state_dict['state_dict'])
    style_encoder.to(torch.device("cuda:0"))
    style_num = len(styles)
    with tqdm.tqdm(total=style_num) as pbar:
        for style in styles:

            style_path = os.path.join(font_dir,style)
            
            available_characters_with_ext_raw = os.listdir(style_path)
            ##
            available_characters_with_ext = []
            with open ("datasets/CFG/chars_800.json",'r') as f:
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
                sty_emb_tmp.extend(style_encoder(image).detach())
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
            torch.save(style_emb,os.path.join(f"{font_dir}_style",f"{style}.pt"))
            pbar.update(1)
            #encodings = torch.stack(encodings).cpu().squeeze()
