import os

import torch

from model_list import model_to_device
from .get_functions import get_save_path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
__all__ = ['load_model']

def load_model(args, model) :
    model_dirs = get_save_path(args)

    load_path = os.path.join(model_dirs, 'model_weights/model_weight_EPOCH{}_fold{}.pth.tar'.format(args.final_epoch, args.current_fold))

    print("Your model is loaded from {}.".format(load_path))
    checkpoint = torch.load(load_path)
    print(".pth.tar keys() =  {}.".format(checkpoint.keys()))

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model = model_to_device(args, model)

    return model