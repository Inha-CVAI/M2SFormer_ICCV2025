import os
import sys

import torch
import torch.optim as optim
__all__ = ['get_device', 'get_optimizer','get_save_path']
def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("You are using \"{}\" device.".format(device))

    return device

def get_optimizer(args, model):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(name)

    if args.optimizer_name == 'SGD' : optimizer = optim.SGD(params=params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer_name == 'Adam' : optimizer = optim.Adam(params=params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == 'AdamW' : optimizer = optim.AdamW(params=params, lr=args.lr, weight_decay=args.weight_decay)
    else : print("Wrong optimizer"); sys.exit()

    return optimizer

def get_save_path(args):
    save_model_path = '{}_{}x{}_{}_{}({}&{})_{}({}_{})'.format(
                                                                  args.train_data_type,
                                                                  str(args.image_size), str(args.image_size),
                                                                  str(args.train_batch_size),
                                                                  args.model_name, args.cnn_backbone,
                                                                  args.transformer_backbone,
                                                                  args.optimizer_name,
                                                                  args.lr,
                                                                  str(args.final_epoch).zfill(3))

    model_dirs = os.path.join(args.save_path, save_model_path)
    if not os.path.exists(os.path.join(model_dirs, 'model_weights')): os.makedirs(os.path.join(model_dirs, 'model_weights'))
    if not os.path.exists(os.path.join(model_dirs, 'test_reports')): os.makedirs(os.path.join(model_dirs, 'test_reports'))
    if not os.path.exists(os.path.join(model_dirs, 'test_reports_per_cases')): os.makedirs(os.path.join(model_dirs, 'test_reports_per_cases'))
    if not os.path.exists(os.path.join(model_dirs, 'plot_results')): os.makedirs(os.path.join(model_dirs, 'plot_results'))

    return model_dirs