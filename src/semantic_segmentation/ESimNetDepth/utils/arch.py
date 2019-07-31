import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from src.semantic_segmentation.ESimNetDepth.models import enet

device = torch.device('cuda')

enetd = enet.ENetDepth(5).to(device)

dummy_data_d_1368 = torch.randn(4, 912, 1368)
dummy_data_d_912 = torch.randn(4, 608, 912)
dummy_data_d_684 = torch.randn(4, 456, 684)
dummy_data_d_456 = torch.randn(4, 304, 456)
dummy_data_d_342 = torch.randn(4, 228, 342)

dummy_data_i_1368 = torch.randn(3, 912, 1368)
dummy_data_i_912 = torch.randn(3, 608, 912)
dummy_data_i_684 = torch.randn(3, 456, 684)
dummy_data_i_456 = torch.randn(3, 304, 456)
dummy_data_i_342 = torch.randn(3, 228, 342)

writer = SummaryWriter('logs/arch')

with SummaryWriter(comment='d1368') as w:
    w.add_graph(enetd, dummy_data_d_1368, verbose=True)

#with SummaryWriter(comment='d912') as w:
#    w.add_graph(enetd, dummy_data_d_912, verbose=True)

eneti = enet.ENet(5)
