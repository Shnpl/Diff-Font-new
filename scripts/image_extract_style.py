import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch 
import torch.distributed as dist

from PIL import Image
import torchvision.transforms as transforms

import json
import math
import random
import torch
#from improved_diffusion.unet import style_encoder_textedit_addskip
from font_classifier import FontClassifier
def resize_image(img, resolution):
    while min(*img.size) >= 2 * resolution:
        img = img.resize(
            tuple(x // 2 for x in img.size), resample=Image.Resampling.BOX
        )

    scale = resolution / min(*img.size)
    img = img.resize(
        tuple(round(x * scale) for x in img.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(img.convert("RGB"))
    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    arr = arr[crop_y: crop_y + resolution, crop_x: crop_x + resolution]
    arr = arr.astype(np.float32) / 127.5 - 1

    transf = transforms.ToTensor()
    img = transf(arr)
    return img

if __name__ == "__main__":
    font_dir = "datasets/CFG/font500_800"
    styles = os.listdir(font_dir)
    #style_encoder = style_encoder_textedit_addskip()
    style_encoder = FontClassifier(use_fc=False)
    style_encoder.eval()
    style_encoder_state_dict = torch.load("models/font_classifier/epoch=24-step=311875.ckpt",map_location='cpu')
    for k in list(style_encoder_state_dict['state_dict'].keys()):
        if k.startswith('model.'):
            style_encoder_state_dict['state_dict'][k[len("model."):]] = style_encoder_state_dict['state_dict'][k]
            del style_encoder_state_dict['state_dict'][k]
    style_encoder.load_state_dict(style_encoder_state_dict['state_dict'])
    style_encoder.to(torch.device("cuda:0"))
    for style in styles:
        style_path = os.path.join(font_dir,style)
        
        available_characters_with_ext_raw = os.listdir(style_path)
        ##
        available_characters_with_ext = []
        with open ("datasets/CFG/seen_characters.json",'r') as f:
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
        #encodings = torch.stack(encodings).cpu().squeeze()
