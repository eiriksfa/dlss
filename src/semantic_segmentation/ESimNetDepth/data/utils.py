import numpy as np
import imageio
import multiprocessing as mp
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as tf
from pathlib import Path

CALC_TYPE = "RGB"


# Image - Label mapping
image_label = {
    (255, 0, 0): 0,
    (0, 0, 255): 1,
    (255, 235, 4): 2,
    (0, 0, 0): 3,
    (255, 255, 255): 4
}
color_map = np.ndarray(shape=(256 * 256 * 256), dtype='int64')
color_map[:] = 0
for rgb, idx in image_label.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    color_map[rgb] = idx

# Label - Image mapping
label_image = {
    0: (255, 0, 0),
    1: (0, 0, 255),
    2: (255, 235, 4),
    3: (0, 0, 0),
    4: (255, 255, 255)
    }
likeys, livalues = zip(*label_image.items())
limap = np.empty((max(likeys) + 1, 3), int)
limap[list(likeys), :] = livalues


def image_to_labels(image):
    image = np.dot(image, np.array([65536, 256, 1], dtype='int64'))
    return color_map[image]


def labels_to_image(labels):
    img = limap[labels, :]
    return img


def output_labels_to_image(labels):
    labels = labels.numpy().transpose((1, 2, 0))
    labels = np.argmax(labels, axis=2)
    image = labels_to_image(labels)
    return image


def get_dirnames(base_path, mode):
    base_path = base_path / mode
    depth_base = base_path / 'depth'
    color_base = base_path / 'scene'
    label_base = base_path / 'color_label'

    images = []

    # Explore the directory tree to get a list of all files
    for color in color_base.iterdir():
        stem = color.stem
        depth = depth_base / (stem + '.png')
        label = label_base / (stem + '.png')
        if color.is_file() and depth.is_file() and label.is_file():
            images.append((color, depth, label))
    return images


def worker(path):
    if CALC_TYPE == "RGB":
        cn = 3
        path = path[0]
        image = cv2.imread(str(path))
        image = image.astype(np.float32) / 255.0
    elif CALC_TYPE == "DEPTH":
        cn = 1
        path = path[1]
        image = cv2.imread(str(path), -2)
        image = image.astype(np.float32) / 65535.0
    else:
        return

    pixel_num = (image.size / cn)
    channel_sum = np.sum(image, axis=(0, 1))
    channel_sum_squared = np.sum(np.square(image), axis=(0, 1))
    return pixel_num, channel_sum, channel_sum_squared


def calc_mean_stddev(path):
    # TODO: Fix for DEpth
    images = get_dirnames(path, 'train')
    if CALC_TYPE == "RGB":
        cn = 3
    elif CALC_TYPE == "DEPTH":
        cn = 1
    else:
        return
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(cn)
    channel_sum_squared = np.zeros(cn)

    pool = mp.Pool()
    results = pool.map(worker, images)

    for (pn, cs, css) in results:
        pixel_num += pn
        channel_sum += cs
        channel_sum_squared += css

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    #
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std


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


def simnet_loader_depth(data_path, depth_path, label_path, width, height, color_mean=[0., 0., 0.], color_std=[1., 1., 1.]):
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
    rgb = rgb[:, :, :3]  # Remove alpha
    # Reshape rgb from H x W x C to C x H x W
    rgb = np.moveaxis(rgb, 2, 0)
    rgb = rgb.astype(np.float32) / 255.0

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
    label = label[:, ::2, ::2]

    return data, label


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:

        w_class = median_freq / freq_class,

    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes

    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq


if __name__ == '__main__':
    m, d = calc_mean_stddev(Path('D:/data/tt2'))
    print(m)
    print(d)
