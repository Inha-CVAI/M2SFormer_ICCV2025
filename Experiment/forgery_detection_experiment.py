import os

import torch
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._base import BaseSegmentationExperiment
from utils import SegmentationMetricsCalculator, load_model, get_save_path


class ForgeryDetectionExperiment(BaseSegmentationExperiment):
    def __init__(self, args):
        super(ForgeryDetectionExperiment, self).__init__(args)

    def fit(self):
        self.print_params()

        print("INFERENCE")
        self._maybe_load_model()

        test_results, total_metrics_dataframe = self.inference()

        return test_results, total_metrics_dataframe

    def _maybe_load_model(self):
        model_dirs = get_save_path(self.args)
        weight_name = f"model_weight_EPOCH{self.args.final_epoch}_fold{self.args.current_fold}.pth.tar"
        weight_path = os.path.join(model_dirs, 'model_weights', weight_name)
        if os.path.exists(weight_path):
            self.model = load_model(self.args, self.model)
        else:
            print(f"WARNING: model weight not found at {weight_path}. Using random initialized weights.")

    def inference(self):
        self.model.eval()

        self.metrics_calculator = SegmentationMetricsCalculator(self.args.metric_list)
        total_metrics_dict = self.metrics_calculator.total_metrics_dict
        metrics_rows = []

        with torch.no_grad():
            for _, data in enumerate(tqdm(self.test_loader)):
                data['epoch'] = 100
                output_dict = self.forward(data)
                if 'prediction' not in output_dict:
                    raise KeyError("Model output does not contain 'prediction'.")

                for target_, output_ in zip(data['target'], output_dict['prediction']):
                    predict = torch.sigmoid(output_).squeeze()
                    metrics_dict = self.metrics_calculator.get_metrics_dict(predict, target_)
                    metrics_rows.append(metrics_dict)
                    for metric in self.metrics_calculator.metrics_list:
                        total_metrics_dict[metric].append(metrics_dict[metric])

        if metrics_rows:
            total_metrics_dataframe = pd.DataFrame(metrics_rows)
            for metric in self.metrics_calculator.metrics_list:
                total_metrics_dict[metric] = np.round(np.mean(total_metrics_dict[metric]), 4)
        else:
            total_metrics_dataframe = pd.DataFrame(columns=self.metrics_calculator.metrics_list)
            for metric in self.metrics_calculator.metrics_list:
                total_metrics_dict[metric] = 0.0

        return total_metrics_dict, total_metrics_dataframe

    def transform_generator(self, mode):
        if mode == 'train':
            transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.args.mean, std=self.args.std),
            ]

            target_transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
            ]
        else:
            transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.args.mean, std=self.args.std)
            ]

            target_transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
            ]

        return transforms.Compose(transform_list), transforms.Compose(target_transform_list)
