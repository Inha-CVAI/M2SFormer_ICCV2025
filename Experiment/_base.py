import os
import random
from datetime import datetime
import sys
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np

from model_list import model_generator, model_to_device
from utils import SegmentationMetricsCalculator, load_model
from utils import (CSVLogger, byte_transform,
                   get_device, get_optimizer, get_save_path)

try:
    from dataset import *
    _DATASET_IMPORT_OK = True
except Exception:
    CASIAV2Dataset = CASIAV1Dataset = COVERAGEDataset = None
    ColumbiaDataset = IMD2020Dataset = CocoGlideDataset = None
    CoMoFoDDataset = IntheWildDataset = MSIDDataset = None
    DIS25KDataset = None
    _DATASET_IMPORT_OK = False

try:
    from PIL import Image
except Exception:
    Image = None


class BaseSegmentationExperiment(object):
    def __init__(self,args):
        super(BaseSegmentationExperiment,self).__init__()

        self.args = args

        self.args.device = get_device()
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.test_loader = self.dataloader_generator()
        self.start, self.end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.inference_time_list = []

        print("STEP2. Load 2D Image Segmentation Model {}...".format(self.args.model_name))
        self.model = model_generator(args)
        self.model = model_to_device(args,self.model)
        self.model = load_model(self.args,self.model)


        self.optimizer = get_optimizer(args,self.model)
        

    def forward(self,data):
        # Single GPU Training

        data = self.cpu_to_gpu(data)

        with torch.cuda.amp.autocast():
            output_dict = self.model(data)

        return output_dict

    def cpu_to_gpu(self, data):
        for key in ['image', 'target']:
            data[key] = data[key].to(self.args.device)

        return data

    def cpu_to_multi_gpu(self,data,devices):
        for key in ['image', 'target']:
            data[key] = data[key].cuda(devices)

        return data

    def current_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
            
        
    def print_params(self):
        print("\ntrain data type : {}".format(self.args.train_data_type))
        print("test data type : {}".format(self.args.test_data_type))

        print("model : {}".format(self.args.model_name))
        print("optimizer : {}".format(self.optimizer))
        print("learning rate : {}".format(self.args.lr))
        print("final epoch : {}".format(self.args.final_epoch))
        print("test batch size : {}".format(self.args.test_batch_size))
        print("image size : ({}, {}, {})".format(self.args.image_size, self.args.image_size, self.args.num_channels))
        print("pytorch_total_params : {}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def worker_init_fn(self, worker_id):
        random.seed(4321 + worker_id)

    def _collect_image_paths(self, root_dir):
        if not root_dir or not os.path.isdir(root_dir):
            return []
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        return [str(p) for p in Path(root_dir).rglob('*') if p.suffix.lower() in exts]

    def dataloader_generator(self):
        train_image_transform, train_target_transform = self.transform_generator('train')
        test_image_transform, test_target_transform = self.transform_generator('test')

        print("Load {} Test Dataset Loader...".format(self.args.test_data_type))
        if self.args.test_data_type == 'CASIAv1': test_dataset = CASIAV1Dataset(self.args, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'CASIAv2': test_dataset = CASIAV2Dataset(self.args, mode='test', transform=train_image_transform, target_transform=train_target_transform)
        elif self.args.test_data_type == 'COVERAGE': test_dataset = COVERAGEDataset(self.args, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'Columbia': test_dataset = ColumbiaDataset(self.args, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'IMD2020': test_dataset = IMD2020Dataset(self.args, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'CocoGlide': test_dataset = CocoGlideDataset(self.args, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'CoMoFoD': test_dataset = CoMoFoDDataset(self.args, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'In_the_Wild': test_dataset = IntheWildDataset(self.args, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'MSID': test_dataset = MSIDDataset(self.args, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        elif self.args.test_data_type == 'DIS25k': test_dataset = DIS25KDataset(self.args, mode='test', transform=test_image_transform, target_transform=test_target_transform)
        else: raise ValueError("Wrong train data type")

        # train_loader = DataLoader(train_dataset,
        #                             batch_size=self.args.train_batch_size,
        #                             shuffle=True,
        #                             num_workers=self.args.num_workers,
        #                             pin_memory=True,
        #                             worker_init_fn=self.worker_init_fn)
        test_loader = DataLoader(test_dataset,
                                    batch_size=self.args.test_batch_size,
                                    shuffle=False,
                                    num_workers=self.args.num_workers,
                                    pin_memory=True,
                                    worker_init_fn=self.worker_init_fn)

        return test_loader