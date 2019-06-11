from pathlib import Path
from src.semantic_segmentation.ESimNetDepth.models.enet import ENet, ENetDepth
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import src.semantic_segmentation.ESimNetDepth.utils.utils as utils
import imageio


MODEL_PATH = Path('Z:/data/blocks/models')
MODEL_NAME = 'TEST'
COLOR_MEAN = [0.496342, 0.466664, 0.440796]
COLOR_STD = [0.277856, 0.286230, 0.291129]
device = torch.device('cuda')
N_CLASSES = 5

COLOR_MAP = [(0, [0, 0, 0]), (1, [0, 0, 255]), (2, [0, 255, 255]), (3, [255, 0, 0])]


def rgb_to_rgbd(rgb, depth):
    normalize = transforms.Normalize(mean=COLOR_MEAN, std=COLOR_STD)
    rgb = np.array(rgb)
    rgb = np.moveaxis(rgb, 2, 0)
    rgb = normalize(torch.Tensor(rgb.astype(np.float32) / 255))
    depth = torch.Tensor(np.array(depth).astype(np.float32) / 65535.0)
    depth = torch.unsqueeze(depth, 0)

    # Concatenate rgb and depth
    data = torch.cat((rgb, depth), 0)
    data = torch.stack([data])
    return data


def pred_to_rgb(pred):
    result = np.zeros([pred.shape[0], pred.shape[1], 3], dtype=np.uint8)
    for m in COLOR_MAP:
        result[(pred == m[0])] = m[1]
    return result


def load_model():
    model = ENetDepth(5).to(device)
    model = nn.DataParallel(model).cuda()
    model = utils.load_checkpoint(model, MODEL_PATH, MODEL_NAME)
    model.eval()
    return model


def predict(rgb, depth, model):
    data = rgb_to_rgbd(rgb, depth)

    with torch.no_grad():
        preds = model(data)

    _, preds = torch.max(preds.data, 1)
    preds = preds.cpu().detach().numpy()
    preds = preds[0]  # 1 result
    return pred_to_rgb(preds)


def main():
    sp = Path('Z:/data/blocks/test/scene/157.png')
    dp = Path('Z:/data/blocks/test/depth/157.png')
    rgb = imageio.imread(sp)
    depth = imageio.imread(dp)
    result = predict(rgb, depth, load_model())
    rp = Path('Z:/data/blocks/result.png')
    imageio.imwrite(rp, result)


if __name__ == '__main__':
    main()
