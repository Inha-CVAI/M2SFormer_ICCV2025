import torch
from torch import nn

def model_generator(args):
    # Forgery Detection Model : SAM-based
    if args.model_name == 'M2SFormer':   from model_list.m2sformer import M2SFormer as model # ICASSP2024

    else: raise ValueError("Wrong model name")


    return model(args)

def _training_configuration(args):

    if args.model_name == 'M2SFormer':    from model_list.m2sformer import _training_config # ICCV2025

    else: raise ValueError("Wrong model name")

    return _training_config(args)

def model_to_device(args,model):

    model = model.to(args.device)
    return model

def pos_weight_calculator(label_mask):
    pos_weight = []

    for target in label_mask:
        num_pos = (target == 1).sum().item()
        num_neg = (target == 0).sum().item()

        pos_weight.append(num_neg / (num_pos + 1e-6))

    pos_weight = torch.tensor(pos_weight).to(label_mask.device).view(-1, 1, 1, 1)

    return pos_weight