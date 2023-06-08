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
    args_dict = vars(args)
    dist_util.setup_dist()
    logger.configure(args_dict["path"])
    
    logger.log("creating model and diffusion...")
    if args_dict["data"]["style_dir"]:
        style_average = True
    else:
        style_average = False
    model, diffusion = create_model_and_diffusion(**args_dict["model"]["params"],style_average = style_average)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args_dict["model"]["schedule_sampler"], diffusion)

    logger.log("creating data loader...")
    data = load_data(
        image_size = args_dict["model"]["params"]["unet_config"]["image_size"],
        **args_dict["data"]        
    )

    logger.log("training...")
    del args_dict["model"]["schedule_sampler"]
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        schedule_sampler=schedule_sampler,
        **args_dict["model"],
        **args_dict["data"]

    ).run_loop()


def create_argparser():
    #defaults = model_and_diffusion_defaults()
    defaults = {}
    path = "logs/logs_20230608"
    with open(os.path.join(path,'train_params.json'),'r') as f:
        modified = json.load(f)
    defaults.update(modified)
    defaults.update({"path":path})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    
    main()
