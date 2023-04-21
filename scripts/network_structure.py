

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
import netron
from tensorboardX import SummaryWriter
def main():
    args = create_argparser().parse_args()

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
    with SummaryWriter("./log", comment="sample_model_visualization") as sw:
        sw.add_graph(model,(
                        torch.zeros((4,3,64,64),dtype=torch.float32),
                        torch.zeros((4),dtype=torch.float32),
                        torch.zeros((4,3,128,128),dtype=torch.float32),
                        torch.zeros((4),dtype=torch.int64),
                        torch.zeros((4,32),dtype=torch.float32)
                    )
        )

def create_argparser():
    defaults = model_and_diffusion_defaults()
    path = "logs/logs_20230416"
    with open(os.path.join(path,'train_params.json'),'r') as f:
        modified = json.load(f)
    defaults.update(modified)
    defaults.update({"path":path})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()



