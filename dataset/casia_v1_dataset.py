import os
import random

import torch
from torch.utils.data import Dataset

from PIL import Image

class CASIAV1Dataset(Dataset):
    def __init__(self, args, mode='test', transform=None, target_transform=None):
        super(CASIAV1Dataset, self).__init__()

        self.args = args
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.dataset_dir = os.path.join(self.args.data_path, 'CASIAv1')

        if not os.path.exists(os.path.join(self.dataset_dir, 'sample_list.txt')):
            print("There is no sample data list. We will make sample data list.")
            self.make_sample_list()

        # Load sample data list
        self.sample_list = []
        with open(os.path.join(self.dataset_dir, 'sample_list.txt'), 'r') as f:
            for line in f:
                self.sample_list.append(line.strip().split())

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_sample, target_sample = self.sample_list[idx][0], self.sample_list[idx][1]

        if 'Sp_S' in image_sample:
            image_path = os.path.join(self.dataset_dir, 'Modified Tp', 'Tp', 'CM', image_sample)
            target_path = os.path.join(self.dataset_dir, 'Gt', 'CM', target_sample)
            image = Image.open(image_path).convert('RGB')
            target = Image.open(target_path).convert('L')

        elif 'Sp_D' in image_sample:
            image_path = os.path.join(self.dataset_dir, 'Modified Tp', 'Tp', 'Sp', image_sample)
            target_path = os.path.join(self.dataset_dir, 'Gt', 'Sp', target_sample)
            image = Image.open(image_path).convert('RGB')
            target = Image.open(target_path).convert('L')

        elif 'Au' in image_sample and self.use_only_tempering == False:
            image_path = os.path.join(self.dataset_dir, 'Au',image_sample)
            image = Image.open(image_path).convert('RGB')
            target = torch.zeros((1, 1024, 1024), dtype=torch.float32)


        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); target = self.target_transform(target)

        target[target >= 0.5] = 1; target[target < 0.5] = 0

        data = {'image': image, 'target': target}

        return data

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def make_sample_list(self):
        self.image_sample_list, self.mask_sample_list = [], []

        # Copy-Move Forgery (459 images)
        # Load sample list CopyMove_groundtruth_list.txt in CASIAv1 dataset
        with open(os.path.join(self.args.dataset_dir, 'Gt', 'CopyMove_groundtruth_list.txt'), 'r') as f:
            for line in f:
                self.mask_sample_list.append(line)
                # image sample without '_gt'
                self.image_sample_list.append(line.split('_gt')[0] + '.jpg')

        assert len(self.image_sample_list) == len(self.mask_sample_list), "The number of image and mask is not same."

        # Splice Forgery (461 images)
        # Load sample list Splicing_groundtruth_list.txt in CASIAv1 dataset
        with open(os.path.join(self.args.dataset_dir, 'Gt', 'Splicing_groundtruth_list.txt'), 'r') as f:
            for line in f:
                self.mask_sample_list.append(line)
                # image sample without '_gt'
                self.image_sample_list.append(line.split('_gt')[0] + '.jpg')

        assert len(self.image_sample_list) == len(self.mask_sample_list), "The number of image and mask is not same."

        print("Dataset Length : {}".format(len(self.image_sample_list)))

        with open(os.path.join(self.args.dataset_dir, 'sample_list.txt'), 'w') as f:
            for image, mask in zip(self.image_sample_list, self.mask_sample_list):
                f.write(f'{image} {mask}')