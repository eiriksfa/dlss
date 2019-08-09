from pathlib import Path
from src.semantic_segmentation.ESimNetDepth.models.enet import ENet, ENetDepth
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import src.semantic_segmentation.ESimNetDepth.utils.utils as utils
import imageio
from src.semantic_segmentation.ESimNetDepth.data.simnetv2 import Transformer as tr
import src.semantic_segmentation.ESimNetDepth.data.utils as dutils


MODEL_PATH = Path('D:/data/test/models')
MODEL_NAME = '1368v51'
COLOR_MEAN = [0.3558754444970275, 0.3683076898847985, 0.3102764670089978]  # [0.496342, 0.466664, 0.440796]
COLOR_STD = [0.1460966414799506, 0.13226721916037815, 0.13365748244322445]  # [0.277856, 0.286230, 0.291129]
device = torch.device('cuda')
N_CLASSES = 5

COLOR_MAP = [(0, [255, 0, 0]), (1, [0, 0, 255]), (2, [255, 235, 0]), (3, [0, 0, 0]), (4, [255, 255, 255])]


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


def predict(rgb, depth, model, transformer):
    rgb = rgb[:, :, :3]  # Remove alpha channel from scene image (should be all 1)
    depth = depth.astype(np.float32) / 65535.0  # Normalize depth
    target = np.zeros(rgb.shape).astype(np.uint8)
    # target = target[:, :, :3]

    data = transformer((rgb, depth, target))

    data = torch.stack([data[0]])

    with torch.no_grad():
        preds = model(data)

    _, preds = torch.max(preds.data, 1)
    preds = preds.cpu().detach().numpy()
    preds = preds[0]  # 1 result
    return pred_to_rgb(preds)


def main():
    sp = Path('D:/data/test/test/scene/152.png')
    dp = Path('D:/data/test/test/depth/152.png')
    transformer = tr(1368, 912, color_mean=COLOR_MEAN, color_std=COLOR_STD, augment=False)
    rgb = imageio.imread(sp)
    depth = imageio.imread(dp)
    result = predict(rgb, depth, load_model(), transformer)
    rp = Path('D:/data/test/test/result.png')
    imageio.imwrite(rp, result)


if __name__ == '__main__':
    main()
