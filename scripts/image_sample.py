"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from PIL import Image
import torchvision.transforms as transforms
import blobfile as bf

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
def main():
    args = create_argparser().parse_args()
    # Generate picture according to model names
    model_name = os.path.basename(args.model_path).split(".")[:-1]
    model_name = "".join(model_name)
    i = 1
    logger_path = os.path.join(os.path.dirname(args.model_path),model_name+"_"+str(i))
    while os.path.exists(logger_path):
        i += 1
        logger_path = os.path.join(os.path.dirname(args.model_path),model_name+"_"+str(i))
    dist_util.setup_dist()
    logger.configure(logger_path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    with open('datasets/CFG/seen_characters.json','r') as f:
        char_800 = json.load(f)
    class_names = []
    for file in [bf.basename(namepath).split("_")[0] for namepath in bf.listdir(args.style_path)]:
        if file.split('.')[0] in char_800:
            class_names.append(file)

    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_label_names = []
    sty_image = []
    sty_stroke = []
    #
    if args.use_stroke:
        stroke_path = "datasets/CFG/new_strokes.json"
        with open(stroke_path,'r') as f:
            strokes = json.load(f)
            stroke_part_list = [
            "㇔",# 点 \u31d4
            "㇐",# 横 \u31d0
            "㇑",# 竖 \u31d1
            "㇓",# 撇 \u31d3
            "㇏",# 捺 \u31cf
            "㇀",# 提 \u31c0
            "𡿨",# 撇点 \ud847\udfe8
            "㇙",# 竖提 \u31d9
            "㇊",# 横折提 \u31ca
            "㇁",# 弯钩 \u31c1
            "㇚",# 竖钩 \u31da
            "㇟",# 竖弯钩 \u31df
            "㇂",# 斜钩 \u31c2
            "㇃",# 卧钩 \u31c3
            "㇖",# 横沟 \u31d6
            "㇆",# 横折钩 \u31c6
            "㇈",# 横折弯钩 \u31c8
            "㇌",# 横撇弯钩 \u31cc
            "㇡",# 横折折折钩 \u31e1
            "㇉",# 竖折折钩 \u31c9
            "㇗",# 竖弯 \u31d7
            "㇍",# 横折弯 \u31cd
            "㇕",# 横折 \u31d5
            "𠃊",# 竖折 \ud840\udcca
            "ㄥ",# 撇折 \u3125
            "㇇",# 横撇 \u31c7
            "㇋",# 横折折撇 \u31cb
            "ㄣ",# 竖折撇 \u3123
            "⺄",# 横斜钩 \u2e84
            "㇞",# 竖折折 \u31de
            "㇅",# 横折折 \u31c5
            "㇎" # 横折折折 \u31ce
            ]
    #
    # TODO There's a bug if the program enters the loop for more than one time
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            # classes = th.randint(
            #     low=0, high=400, size=(args.batch_size,), device=dist_util.dev()
            # )
            classes = th.tensor(range(args.batch_size),device=dist_util.dev())
            #classes = th.tensor(np.array([5209,4777,218,1964,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])).to(dist_util.dev())
            for index in classes:
                for key, value in sorted_classes.items():
                    if int(index) == value:
                        all_label_names.append(key)
            
            for imgname in all_label_names:
                image = Image.open(os.path.join(args.style_path, imgname))
                image.load()
                image = resize_image(image,128)
                sty_image.append(image.unsqueeze(0))
                if args.use_stroke:
                    char_strokes = strokes[imgname.split(".")[0]]
                    stroke_count_dict = dict([(stroke_part,0) for stroke_part in stroke_part_list])
                    for stroke in char_strokes:
                        stroke_count_dict[stroke] += 1
                    style_stroke_list = []
                    for stroke in stroke_part_list:
                        style_stroke_list.append(stroke_count_dict[stroke])
                    sty_stroke.append(np.array(style_stroke_list,dtype=np.float32))
            
            sty_image = th.cat(sty_image, dim=0).to(dist_util.dev())
            
            model_kwargs["y"] = classes
            if args.use_stroke:
                sty_stroke = th.tensor(sty_stroke).to(dist_util.dev())
                model_kwargs["stroke"] = sty_stroke
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            sty_image,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            images = []
            for i in range(args.num_samples):
                img = Image.fromarray(arr[i])
                images.append(img)
            image_num = len(images)
            width_num = 8
            height_num = math.ceil(image_num/width_num)*2
            target_shape = (width_num*64,height_num*64)
            background = Image.new("RGB",target_shape,(0,0,0,))
            location = (0,0)
            
            for image in images:
                background.paste(image,location)
                location = (location[0]+64,location[1]+0)
                if location[0] >= 64*8:
                    location = (0,location[1]+64*2)
            location = (0,64)
            for label in all_label_names:
                label = label.split(".")[0]
                gt_image = Image.open(os.path.join(args.style_path,label.split('.')[0]+'.png'))
                gt_image = gt_image.resize((64,64))
                background.paste(gt_image,location)
                location = (location[0]+64,location[1]+0)
                if location[0] >= 64*8:
                    location = (0,location[1]+64*2)
            background.save(f"{out_path.split('.')[0]}.jpg")
                #label = all_label_names[i]
                #label = label.split(".")[0]
                #with open(f"{path}/{label}.jpg",'w') as f:
                #    img.save(f)
            np.savez(out_path, arr, label_arr)

        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults=model_and_diffusion_defaults() 
    path = "logs/logs_20230421"
    with open (os.path.join(path,'val_params.json'),"r") as f:    
        modified = json.load(f)
    defaults.update(modified)
    defaults.update({"path":path})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def resize_image(img, resolution):
    while min(*img.size) >= 2 * resolution:
        img = img.resize(
            tuple(x // 2 for x in img.size), resample=Image.BOX
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
