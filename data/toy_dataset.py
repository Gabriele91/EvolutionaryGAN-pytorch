
import os
import os.path
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
from data.base_dataset import BaseDataset, get_transform
import random
import torch

class ToyDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--toy_name', type=str, default='gaussian25',  help='toy name [gaussian25, gaussian10]')
        return parser

    @staticmethod
    def toy_dataset(DATASET='gaussians8', size=256):
        if DATASET == 'gaussian25':
            dataset = []
            for i in range(int(500/25)):
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        point = np.random.randn(2)*0.05
                        point[0] += 2*x
                        point[1] += 2*y
                        dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            np.random.shuffle(dataset)
            dataset /= 2.828  # stdev
        elif DATASET == 'gaussian8':
            scale = 2.
            centers = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1./np.sqrt(2), 1./np.sqrt(2)),
                (1./np.sqrt(2), -1./np.sqrt(2)),
                (-1./np.sqrt(2), 1./np.sqrt(2)),
                (-1./np.sqrt(2), -1./np.sqrt(2))
            ]
            centers = [(scale*x, scale*y) for x, y in centers]
            dataset = []
            for i in range(size):
                point = np.random.randn(2)*.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev
        return dataset

    def __init__(self, transform=None, target_transform=None, toy_name="gaussian25",  **kwargs):
        self.true_dist = ToyDataset.toy_dataset(toy_name)
        np.random.shuffle(self.true_dist)
        self.true_dist /= 2.828  # stdev
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img = torch.Tensor(self.true_dist[index])
        target = np.array([1])
        #print(img.size(), target)
        return {'image': img, "target": int(target[0])}

    def __len__(self):
        return len(self.true_dist)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
