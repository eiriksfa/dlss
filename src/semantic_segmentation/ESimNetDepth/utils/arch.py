import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from src.semantic_segmentation.ESimNetDepth.models import enet
import torch.optim as optim
import src.semantic_segmentation.ESimNetDepth.utils.utils as utils
from pathlib import Path
from src.semantic_segmentation.ESimNetDepth.data.simnetv2 import SimNet as dataset
import torch.utils.data as data

SAVE_DIR = Path('D:/data/models')
ROOT_DIR = Path('D:/data')
NAME = '456v1'
IMAGE_HEIGHT = 304
IMAGE_WIDTH = 456

COLOR_MEAN = [0.496342, 0.466664, 0.440796]
COLOR_STD = [0.277856, 0.286230, 0.291129]
LOAD_DEPTH = True

device = torch.device('cuda')

model = enet.ENetDepth(5).to(device)

# model = torch.nn.DataParallel(model).cuda()

# model.eval()

test_set = dataset(ROOT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, mode='test', color_mean=COLOR_MEAN, color_std=COLOR_STD, load_depth=LOAD_DEPTH)
test_loader = data.DataLoader(test_set, batch_size=1, shuffle=True)

# Initialize a optimizer just so we can retrieve the model from the
# checkpoint
# optimizer = optim.Adam(model.parameters())

# Load the previoulsy saved model state to the ENet model
# model = utils.load_checkpoint(model, SAVE_DIR, NAME)

# dummy_data = torch.randn(4, 912, 1368)
# dummy_data = torch.randn(4, 608, 912)
# dummy_data = torch.randn(4, 456, 684)
# dummy_data = torch.randn(4, 304, 456)
# dummy_data = torch.randn(4, 228, 342)

# dummy_data = torch.randn(3, 912, 1368)
# dummy_data = torch.randn(3, 608, 912)
# dummy_data = torch.randn(3, 456, 684)
# dummy_data = torch.randn(3, 304, 456)
# dummy_data = torch.randn(3, 228, 342)

writer = SummaryWriter('D:/data/logs/arch')

epoch_loss = 0.0
avgTime = 0.0
numTimeSteps = 0
n = 0
for step, batch_data in enumerate(test_loader):
    # Get the inputs and labels
    inputs = batch_data[0].to(device)
    # labels = batch_data[1].long().to(device)
    n += 1

    with torch.no_grad():
        # Forward propagation
        # outputs = model(inputs)
        if n == 1:
            with SummaryWriter(comment='d456') as w:
                w.add_graph(model, inputs, verbose=True)
        exit()


