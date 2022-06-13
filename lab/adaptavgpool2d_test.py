import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

from collections import OrderedDict as od

m = nn.AdaptiveAvgPool2d((18,18))
input = torch.randn((1,64,8,9))
output = m(input)
print(output.size())