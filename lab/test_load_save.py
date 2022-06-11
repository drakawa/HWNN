import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

from collections import OrderedDict as od

class GraphLayer(nn.Module):
    def __init__(self, channels, n_steps, G: nx.Graph):
        super(GraphLayer, self).__init__()

        self.n_steps = n_steps
        self.nbrs_odict = od()
        self.nodes = list(sorted(G.nodes()))

        for n in self.nodes:
            self.nbrs_odict[n] = list(sorted(G[n]))

        self.layers = nn.ModuleDict({})
        for n in sorted(self.nodes):
            self.layers["{}".format(n)] = RecNodeLayer(n, self.nbrs_odict[n], channels)

    def forward(self, y):
        # step 0
        outputs = od()
        for n in self.nodes:
            outputs[n] = self.layers["{}".format(n)](y, 0)

        # step 1 ~ (n_steps-1)
        for step in range(1, self.n_steps):
            new_outputs = od()
            for n in self.nodes:
                # new_y = torch.cat(tuple([outputs[n]] + [outputs[i] for i in self.nbrs_odict[n]]), dim=-1)
                new_y = torch.stack(list([outputs[n]] + [outputs[i] for i in self.nbrs_odict[n]]), dim=-1)
                print("new_y.size():", new_y.size())
                new_outputs[n] = self.layers["{}".format(n)](new_y, step)
            outputs = new_outputs

        y = torch.mean(y, dim=-1)
        return y

class RecNodeLayer(nn.Module):
    def __init__(self, node_id, neighbors, channels, stride=1):
        super(RecNodeLayer, self).__init__()
        self.node_id = node_id
        self.neighbors = neighbors

        self.params = nn.ParameterDict({})

        self.params["input"] = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.params["{}".format(self.node_id)] = nn.Parameter(torch.zeros(1, requires_grad=True))    
        for nbr in self.neighbors:
            self.params["{}".format(nbr)] = nn.Parameter(torch.zeros(1, requires_grad=True))

        # self.agg_weight = torch.cat((self.params["{}".format(self.node_id)], *[self.params["{}".format(nbr)] for nbr in self.neighbors]), dim=0)
        # print("agg_weight:")
        # print(self.agg_weight)

        self.conv = SeparableConv2d(channels,channels, kernels_per_layer=1, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(channels)

        pass

    def forward(self, y, step):
        if step == 0:
            # y = torch.matmul(y, torch.sigmoid(self.params["input"]))
            y *= torch.sigmoid(self.params["input"])
        else:
            y = torch.matmul(y, torch.sigmoid(torch.cat((self.params["{}".format(self.node_id)], *[self.params["{}".format(nbr)] for nbr in self.neighbors]), dim=-1)))

        identity = y.clone()

        y = F.relu(y)
        y = self.conv(y)
        y = self.bn(y)
        y += identity

        # y = self.conv(y)
        # y = self.bn(y)
        # y = F.relu(y)
        # y += identity

        # y = self.conv(y)
        # y = self.bn(y)
        # y += identity
        # y = F.relu(y)

        return y
        

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer, kernel_size=1, stride=1, padding=0):
        super(SeparableConv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels*kernels_per_layer, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels*kernels_per_layer, out_channels, kernel_size=1, stride=1, padding=0, groups=1)
    
    def forward(self,x):
        # print(x.size())
        # print("x.device", x.device)
        x = self.conv1(x)
        # print(x.size())
        x = self.pointwise(x)
        # print(x.size())
        return x

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TheModelClass2(TheModelClass):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
        self.linears2 = nn.ModuleDict({"linear_{}".format(i):nn.Linear(10, 10) for i in range(10)})

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print(model.state_dict()["conv1.bias"])

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

PATH = "./test_load_save.pth"
torch.save(model.state_dict(), PATH)

model = TheModelClass()
model.load_state_dict(torch.load(PATH))
model.eval()

model = TheModelClass2()
print(model.state_dict()["conv1.bias"])
print(model.state_dict()["conv3.bias"])

model.load_state_dict(torch.load(PATH), strict=False)
model.eval()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print(model.state_dict()["conv1.bias"])
print(model.state_dict()["conv3.bias"])

net = RecNodeLayer(0,[1,2,3],8,1)
print("RecNodeLayer's state_dict:")
print(net.state_dict())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.randn(64,8,16,16)

net = net.to(device)
x = x.to(device)

print(x.size())
y = net(x, 0)
print(y.size())

x = torch.randn(64,8,16,16,4)

net = net.to(device)
x = x.to(device)

print("params:")
for param in net.parameters():
    print(param)

print(x.size())
y = net(x, 1)
print(y.size())

net = GraphLayer(64,5,nx.random_regular_graph(3,8,1))
x = torch.randn(7,64,16,16)

net = net.to(device)
x = x.to(device)

params = 0
for p in net.parameters():
    if p.requires_grad:
        params += p.numel()
        
print(params)  # 121898

y = net(x)
