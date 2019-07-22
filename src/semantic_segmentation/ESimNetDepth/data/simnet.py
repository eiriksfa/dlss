from collections import OrderedDict
import torch.utils.data as data
import src.semantic_segmentation.ESimNetDepth.data.utils as utils  # from src.semantic_segmentation.ESimNetDepth.data
from pathlib import Path


class SimNet(data.Dataset):

    def __init__(self, root_dir, mode='train', color_mean=[0., 0., 0.], color_std=[1., 1., 1.], load_depth=True):
        self.root_dir = root_dir
        self.mode = mode
        self.color_mean = color_mean
        self.color_std = color_std
        self.load_depth = load_depth
        self.length = 0
        self.loader = utils.simnet_loader_depth if load_depth else utils.simnet_loader
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.color_encoding = self.get_color_encoding()

        print(self.mode)

        if self.mode.lower() == 'train':
            images = utils.get_dirnames(self.root_dir, 'train')
            self.train_data = images
            self.length = len(images)

        elif self.mode.lower() == 'val':
            # Get val data and labels filepaths
            images = utils.get_dirnames(self.root_dir, 'val')
            self.val_data = images
            self.length = len(images)

        elif self.mode.lower() == 'test' or self.mode.lower() == 'inference':
            # Get test data and labels filepaths
            images = utils.get_dirnames(self.root_dir, 'test')
            self.test_data = images
            self.length = len(images)
        else:
            raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test')

    def __getitem__(self, item):
        """ Returns element at index in the dataset.

        Args:
        - item (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth of the image

        """

        if self.load_depth is True:

            if self.mode.lower() == 'train':
                (data_path, depth_path, label_path) = self.train_data[item]
            elif self.mode.lower() == 'val':
                (data_path, depth_path, label_path) = self.val_data[item]
            elif self.mode.lower() == 'test':
                (data_path, depth_path, label_path) = self.test_data[item]
            else:
                raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test')

            rgbd, label = self.loader(data_path, depth_path, label_path, self.color_mean, self.color_std)

            return rgbd, label

        else:

            if self.mode.lower() == 'train':
                (data_path, _, label_path) = self.train_data[item]
            elif self.mode.lower() == 'val':
                (data_path, _, label_path) = self.val_data[item]
            elif self.mode.lower() == 'test' or self.mode.lower() == 'inference':
                (data_path, _, label_path) = self.test_data[item]
            else:
                raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test')

            img, label = self.loader(data_path, label_path, self.color_mean, self.color_std)

            return img, label

    def __len__(self):
        return self.length

    def get_color_encoding(self):
        return OrderedDict([
            ('unsafe', (255, 0, 0)),
            ('safe', (0, 0, 255)),
            ('msafe', (0, 255, 255)),
            ('background', (0, 0, 0)),
            ('unlabeled', (255, 255, 255))
        ])
