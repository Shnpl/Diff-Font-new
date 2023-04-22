"""
Train a diffusion model on images.
"""

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

import torch
import os
import json

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.pretrained_dict)
    pretrained_dict = pretrained_dict['netStyleEncoder']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'style_encoder.' + k in model_dict}
    style_encoder_dict = {}
    for k, v in pretrained_dict.items():
        style_encoder_dict.update({'style_encoder.' + k: v})
    model_dict.update(style_encoder_dict)
    model.load_state_dict(model_dict)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    for param in model.style_encoder.parameters():
        param.requires_grad = False

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        use_stroke = args.use_stroke
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = model_and_diffusion_defaults()
    path = "logs/logs_20230421"
    with open(os.path.join(path,'train_params.json'),'r') as f:
        modified = json.load(f)
    defaults.update(modified)
    defaults.update({"path":path})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
