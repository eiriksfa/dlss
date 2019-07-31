from collections import OrderedDict
import torch.utils.data as data
import src.semantic_segmentation.ESimNetDepth.data.utils as utils
from pathlib import Path
import numpy as np
import imageio
import torch
from torchvision.transforms import functional as func
from matplotlib import pyplot as plt


class Transformer:

    def __init__(self, width, height, depth=True, augment=True):
        self.width = width
        self.height = height
        self.augment = augment
        self.depth = depth

        self.transformations = [self._to_pil,
                                self._resize]

        augmentations = [self._adjust_brightness,
                         self._adjust_contrast,
                         self._adjust_gamma,
                         self._adjust_saturation,
                         self._rotate]

        finalize = [self._to_label,
                    self._to_tensor,
                    self._combine]

        if augment:
            self.transformations = self.transformations + augmentations

        self.transformations = self.transformations + finalize

    def __call__(self, imageset):
        for tr in self.transformations:
            imageset = tr(imageset)
        return imageset

    @staticmethod
    def _to_label(imgset):
        return imgset[0], imgset[1], utils.image_to_labels(imgset[2])

    @staticmethod
    def _to_pil(imgset):
        return func.to_pil_image(imgset[0]), func.to_pil_image(imgset[1]), func.to_pil_image(imgset[2])

    def _resize(self, imgset):
        size = (self.height, self.width)
        return func.resize(imgset[0], size), func.resize(imgset[1], size), func.resize(imgset[2], size)

    @staticmethod
    def _to_tensor(imgset):
        return func.to_tensor(imgset[0]), func.to_tensor(imgset[1]), imgset[2]

    @staticmethod
    def _adjust_brightness(imgset):
        chance = 0.2
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.5, high=1.5)
        return func.adjust_brightness(imgset[0], v), imgset[1], imgset[2]

    @staticmethod
    def _adjust_contrast(imgset):
        chance = 0.25
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.2, high=1.8)
        return func.adjust_contrast(imgset[0], v), imgset[1], imgset[2]

    @staticmethod
    def _adjust_gamma(imgset):
        chance = 0.25
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.2, high=1.8)
        return func.adjust_gamma(imgset[0], v), imgset[1], imgset[2]

    @staticmethod
    def _adjust_saturation(imgset):
        chance = 0.25
        if np.random.random() > chance:
            return imgset
        v = np.random.uniform(low=0.5, high=1.5)
        return func.adjust_saturation(imgset[0], v), imgset[1], imgset[2]

    @staticmethod
    def _rotate(imgset):
        chance = 0.4
        if np.random.random() > chance:
            return imgset
        v = np.random.randint(-30, 30)
        return func.rotate(imgset[0], v), func.rotate(imgset[1], v), func.rotate(imgset[2], v)

    def _combine(self, imgset):
        img = torch.cat((imgset[0], imgset[1]), 0) if self.depth else imgset[0]
        return img, imgset[2]


class SimNet(data.Dataset):

    def __init__(self, root_dir, width, height, mode='train', color_mean=[0., 0., 0.], color_std=[1., 1., 1.], load_depth=True):
        self.root_dir = root_dir
        self.width = width
        self.height = height
        self.mode = mode
        self.color_mean = color_mean
        self.color_std = color_std
        self.load_depth = load_depth
        self.transformer = Transformer(width=width, height=height, depth=load_depth, augment=False)

        self.data = utils.get_dirnames(self.root_dir, mode.lower())
        self.length = len(self.data)

    def __getitem__(self, item):

        (data_path, depth_path, label_path) = self.data[item]
        scene = self._get_image(data_path)
        scene = scene[:, :, :3]  # Remove alpha channel from scene image (should be all 1)
        depth = self._get_image(depth_path)
        depth = depth.astype(np.float32) / 65535.0  # Normalize depth
        target = self._get_image(label_path)
        target = target[:, :, :3]  # Remove alpha channel from target image (should be all 1)

        return self.transformer((scene, depth, target))

    def __len__(self):
        return self.length

    @staticmethod
    def _get_image(path):
        img = imageio.imread(path)
        return img

    def get_color_encoding(self):
        return OrderedDict([
            ('unsafe', (255, 0, 0)),
            ('safe', (0, 0, 255)),
            ('msafe', (255, 235, 4)),
            ('background', (0, 0, 0)),
            ('unlabeled', (255, 255, 255))
        ])


def main():
    p = Path('D:/data/test')
    ts = SimNet(p, 684, 456)
    img = ts.__getitem__(4)
    s = img[0].numpy().transpose((1, 2, 0))
    plt.imshow(s[:, :, :3])
    plt.show()
    d = s[:, :, 3:] * 65535
    d = d.astype(np.uint16)
    imageio.imwrite('D:/data/test/test.png', d)
    #plt.imshow(d)
    #plt.show()
    #labels = img[1].transpose((1, 2, 0))
    #labels = np.argmax(img[1], axis=2)
    result = utils.labels_to_image(img[1])
    plt.imshow(result)
    plt.show()


if __name__ == '__main__':
    main()
