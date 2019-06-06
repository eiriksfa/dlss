import numpy as np
import imageio

import torch
import torchvision.transforms as transforms


def get_dirnames(base_path, mode):
    base_path = base_path / mode
    depth_base = base_path / 'depth'
    color_base = base_path / 'scene'
    label_base = base_path / 'label'

    images = []

    # Explore the directory tree to get a list of all files
    for color in color_base.iterdir():
        stem = color.stem
        depth = depth_base / (stem + '.png')
        label = label_base / (stem + '.png')
        if color.is_file() and depth.is_file() and label.is_file():
            images.append((color, depth, label))
    return images


def simnet_loader(data_path, label_path, color_mean=[0., 0., 0.], color_std=[1., 1., 1.]):
    """Loads a sample and label image given their path as PIL images. (nyu40 classes)

    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - label_path (``string``): The filepath to the ground-truth image.
    - color_mean (``list``): R, G, B channel-wise mean
    - color_std (``list``): R, G, B channel-wise stddev

    Returns the image and the label as PIL images.

    """

    # Load image
    data = np.array(imageio.imread(data_path))
    # Reshape data from H x W x C to C x H x W
    data = np.moveaxis(data, 2, 0)
    # Define normalizing transform
    normalize = transforms.Normalize(mean=color_mean, std=color_std)
    # Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
    data = normalize(torch.Tensor(data.astype(np.float32) / 255.0))

    # Load label
    label = np.array(imageio.imread(label_path)).astype(np.uint8)

    return data, label


def simnet_loader_depth(data_path, depth_path, label_path, color_mean=[0., 0., 0.], color_std=[1., 1., 1.]):
    """Loads a sample and label image given their path as PIL images. (nyu40 classes)

    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - depth_path (``string``): The filepath to the depth png.
    - label_path (``string``): The filepath to the ground-truth image.
    - color_mean (``list``): R, G, B channel-wise mean
    - color_std (``list``): R, G, B channel-wise stddev

    Returns the image and the label as PIL images.

    """

    # Load image
    rgb = np.array(imageio.imread(data_path))
    # Reshape rgb from H x W x C to C x H x W
    rgb = np.moveaxis(rgb, 2, 0)
    # Define normalizing transform
    normalize = transforms.Normalize(mean=color_mean, std=color_std)
    # Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
    rgb = normalize(torch.Tensor(rgb.astype(np.float32) / 255.0))

    # Load depth
    depth = torch.Tensor(np.array(imageio.imread(depth_path)).astype(np.float32) / 65535.0)
    depth = torch.unsqueeze(depth, 0)

    # Concatenate rgb and depth
    data = torch.cat((rgb, depth), 0)

    # Load label
    label = np.array(imageio.imread(label_path)).astype(np.uint8)

    return data, label
