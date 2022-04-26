import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
import numpy as np

################ ResNet ####################
class Block_ppc(nn.Module):
    def __init__(self, in_channels): # PxP, C -> PxP, C
        super(Block_ppc, self).__init__()
        assert in_channels % 4 == 0, print("invalid_in_channels", in_channels)

        in_channels_div4 = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, in_channels_div4, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels_div4)

        self.conv2 = nn.Conv2d(in_channels_div4, in_channels_div4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels_div4)

        self.conv3 = nn.Conv2d(in_channels_div4, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.size())
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print(x.size())

        x = self.conv3(x)
        x = self.bn3(x)
        # print(x.size())

        x += identity
        x = self.relu(x)
        # print(x.size())

        return x
        
class Block_ppc4(nn.Module):
    def __init__(self, in_channels): # PxP, C -> PxP, C*4
        super(Block_ppc4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        in_channels_mul4 = in_channels * 4
        self.conv3 = nn.Conv2d(in_channels, in_channels_mul4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(in_channels_mul4)

        self.relu = nn.ReLU()

        self.identity_downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels_mul4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels_mul4),
        )

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.size())
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print(x.size())

        x = self.conv3(x)
        x = self.bn3(x)
        # print(x.size())

        x += self.identity_downsample(identity)
        x = self.relu(x)
        # print(x.size())

        return x
        
class Block_phphc2(nn.Module):
    def __init__(self, in_channels): # PxP, C -> (P/2)x(P/2), C*2
        super(Block_phphc2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        in_channels_mul2 = in_channels * 2
        self.conv3 = nn.Conv2d(in_channels, in_channels_mul2, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(in_channels_mul2)

        self.relu = nn.ReLU()

        self.identity_downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels_mul2, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels_mul2),
        )

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.size())
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print(x.size())

        x = self.conv3(x)
        x = self.bn3(x)
        # print(x.size())

        x += self.identity_downsample(identity)
        x = self.relu(x)
        # print(x.size())

        return x
        
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        in_channels = 3
        out_channels = 64
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.channels = out_channels

        self.conv2 = nn.Sequential(
            Block_ppc4(in_channels=self.channels),
            Block_ppc(in_channels=self.channels*4),
            Block_ppc(in_channels=self.channels*4),
        )
        
        self.conv3 = nn.Sequential(
            Block_phphc2(in_channels=self.channels*4),
            Block_ppc(in_channels=self.channels*8),
            Block_ppc(in_channels=self.channels*8),
            Block_ppc(in_channels=self.channels*8),
        )
        
        self.conv4 = nn.Sequential(
            Block_phphc2(in_channels=self.channels*8),
            Block_ppc(in_channels=self.channels*16),
            Block_ppc(in_channels=self.channels*16),
            Block_ppc(in_channels=self.channels*16),
            Block_ppc(in_channels=self.channels*16),
            Block_ppc(in_channels=self.channels*16),
        )
        
        self.conv5 = nn.Sequential(
            Block_phphc2(in_channels=self.channels*16),
            Block_ppc(in_channels=self.channels*32),
            Block_ppc(in_channels=self.channels*32),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.channels*32, num_classes)
        
    def forward(self, x):

        ### CONV1 ####
        # print("CONV1")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.size())

        # print("CONV2")
        ### CONV2 ####
        x = self.maxpool(x)
        x = self.conv2(x)
        # print(x.size())

        # print("CONV3")
        ### CONV3 ###
        x = self.conv3(x)
        # print(x.size())
        
        # print("CONV4")
        ### CONV4 ###
        x = self.conv4(x)
        # print(x.size())
        
        # print("CONV5")
        ### CONV5 ###
        x = self.conv5(x)
        # print(x.size())
        
        x = self.avgpool(x)
        # print(x.size())
        x = x.reshape(x.shape[0], -1)
        # print(x.size())
        x = self.fc(x)
        # print(x.size())

        return x
        
################ Randomly Wired Neural Networks ####################
class RWNN(nn.Module):
    def __init__(self, chn, cls, im, Gs):
        super(RWNN, self).__init__()

        dropout_rate = 0.2
        self.conv1 = Conv2d_CB(im, chn // 2, 3, 2, 1)
        self.conv2 = Conv2d_RCB(chn // 2, chn, kernel_size=3, stride=2, padding=1)
        self.conv3 = DAGLayer(chn, chn, Gs[0])
        self.conv4 = DAGLayer(chn, chn * 2, Gs[1])
        self.conv5 = DAGLayer(chn * 2, chn * 4, Gs[2])

        self.class_conv = Conv2d_RCB(chn * 4, 1280)
        self.class_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.class_dropout = nn.Dropout(dropout_rate)
        self.class_fc = nn.Linear(1280, cls)

    def forward(self, y):
        # print(y.size())
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)

        y = self.class_conv(y)
        y = self.class_avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.class_dropout(y)
        y = self.class_fc(y)
        y = F.log_softmax(y, dim=1)

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

class Conv2d_CB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(Conv2d_CB,self).__init__()

        # self.conv = SeparableConv2d(in_channels, out_channels, kernels_per_layer=1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        # print(x.size())
        return x

class Conv2d_RCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(Conv2d_RCB,self).__init__()

        self.relu = nn.ReLU()
        # self.conv = SeparableConv2d(in_channels, out_channels, kernels_per_layer=1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        # print(x.size())
        return x

class DAGLayer(nn.Module):
    def __init__(self, in_channels, out_channels, G: nx.Graph):
        super(DAGLayer, self).__init__()
        in_edges_dict = dict()
        self.has_in = dict()
        self.has_out = dict()
        for n in G.nodes():
            n_nbrs = np.array(G[n])
            in_edges_dict[n] = n_nbrs[n_nbrs < n]
            self.has_in[n] = len(in_edges_dict[n]) > 0
            out_edges = n_nbrs[n_nbrs > n]
            self.has_out[n] = len(out_edges) > 0

        # print(in_edges_dict)
        # print(self.has_in)
        self.layers = nn.ModuleList()
        for n in sorted(G.nodes()):
            if not self.has_in[n]:
                self.layers.append(NodeLayer_in(in_channels, out_channels))
            else:
                self.layers.append(NodeLayer(out_channels, out_channels, in_edges_dict[n]))

        # print(self.layers)

    def forward(self, y):
        y_orig = y.clone()
        # print("forward of DAG_layer:", y.size())
        for n, layer in enumerate(self.layers):
            # print(n, y.size())
            if not self.has_in[n]:
                # print("not has_in", n)
                new_y = layer(y_orig)
                if n == 0:
                    y = torch.unsqueeze(new_y, -1)
                else:
                    new_y = torch.unsqueeze(new_y, -1)
                    y = torch.cat((y, new_y), -1)
            else:
                new_y = layer(y)
                new_y = torch.unsqueeze(new_y, -1)
                # print(new_y.size())
                y = torch.cat((y, new_y), -1)
        y = y[...,[k for k,v in sorted(self.has_out.items()) if not v]]
        # print(self.has_out)
        # print(y.size())
        y = torch.mean(y, dim=-1)
        return y

class NodeLayer_in(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NodeLayer_in, self).__init__()
        in_degree = 1
        stride = 2
        # self.node_op = NodeOp(in_degree, in_channels, out_channels, stride)
        self.node_op = NodeOp_in(in_channels, out_channels, stride)

    def forward(self, y):
        # y = torch.unsqueeze(y, -1)
        y = self.node_op(y)
        return y

class NodeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, in_edge_list):
        super(NodeLayer, self).__init__()
        self.in_edge_list = in_edge_list
        in_degree = len(self.in_edge_list)
        stride = 1
        self.node_op = NodeOp(in_degree, in_channels, out_channels, stride)

    def forward(self, y):
        # print(y.size())
        # print(self.in_edge_list)
        y = y[...,[i for i in self.in_edge_list]]
        y = self.node_op(y)
        return y

class NodeOp_in(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(NodeOp_in, self).__init__()
        self.conv = SeparableConv2d(in_channels, out_channels, kernels_per_layer=1, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, y):
        # print(y.device)
        # y: [B, C, N, M, in_degree]
        # y = torch.matmul(y, self.agg_weight) # [B, C, N, M]
        # y = torch.squeeze(y, -1)
        y = F.relu(y) # [B, C, N, M]
        y = self.conv(y) # [B, C_out, N, M]
        y = self.bn(y) # [B, C_out, N, M]
        return y

class NodeOp(nn.Module):
    def __init__(self, in_degree, in_channels, out_channels, stride):
        super(NodeOp, self).__init__()
        self.single_in = in_degree == 1
        if not self.single_in:
            self.agg_weight = nn.Parameter(torch.zeros(in_degree, requires_grad=True))
        # print(self.agg_weight.device)
        self.conv = SeparableConv2d(in_channels, out_channels, kernels_per_layer=1, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, y):
        # print(y.device)
        # y: [B, C, N, M, in_degree]
        if self.single_in:
            y = torch.squeeze(y, -1)
        else:
            y = torch.matmul(y, torch.sigmoid(self.agg_weight)) # [B, C, N, M]
        # y = torch.matmul(y, self.agg_weight) # [B, C, N, M]
        y = F.relu(y) # [B, C, N, M]
        y = self.conv(y) # [B, C_out, N, M]
        y = self.bn(y) # [B, C_out, N, M]
        return y

if __name__ == "__main__":
    print("Block_ppc")
    x = torch.randn(4,256,56,56)
    print(x.size())
    net = Block_ppc(in_channels=256)
    y = net(x)
    print(y.size())

    print("Block_ppc4")
    x = torch.randn(4,256,56,56)
    print(x.size())
    net = Block_ppc4(in_channels=256)
    y = net(x)
    print(y.size())
    
    print("Block_phphc2")
    x = torch.randn(4,256,56,56)
    print(x.size())
    net = Block_phphc2(in_channels=256)
    y = net(x)
    print(y.size())
    
    print("ResNet50")
    x = torch.randn(1,3,224,224)
    print(x.size())
    net = ResNet50(num_classes=1000)
    params = 0
    for p in net.parameters():
        # print(p)
        if p.requires_grad:
            params += p.numel()
            
    print(params)  # 121898
    

    y = net(x)
    print(y.size())

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.randn(4,256,56,56)
    print(x.size())
    net = SeparableConv2d(in_channels=256, out_channels=256, kernels_per_layer=1, kernel_size=3, padding=1)
    net = net.to(device)
    # print(list(net.parameters()))
    x = x.to(device)
    y = net(x)
    print(y.size())

    x = torch.randn(4,3,32,32)
    print(x.size())
    net = SeparableConv2d(in_channels=3, out_channels=64, kernels_per_layer=1, kernel_size=3, stride=2, padding=1)
    net = net.to(device)
    # print(list(net.parameters()))
    x = x.to(device)
    y = net(x)
    print(y.size())

    print("Conv2d_RCB:")
    x = torch.randn(4,64,32,32)
    print(x.size())
    net = Conv2d_RCB(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
    net = net.to(device)
    # print(list(net.parameters()))
    x = x.to(device)
    y = net(x)
    print(y.size())

    print("Conv2d_CB:")
    x = torch.randn(100,3,32,32)
    print(x.size())
    net = Conv2d_CB(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
    net = net.to(device)
    # print(list(net.parameters()))
    x = x.to(device)
    y = net(x)
    print(y.size())

    x = torch.randn(4,256,56,56,3)
    print(x.size())
    net = NodeOp(in_degree=3, in_channels=256, out_channels=256, stride=1)
    net = net.to(device)
    # print(list(net.parameters()))
    x = x.to(device)
    y = net(x)
    print(y.size())

    x = torch.randn(4,256,56,56) # 1 input node
    net = NodeLayer_in(in_channels=256, out_channels=256)
    net = net.to(device)
    x = x.to(device)
    y = net(x)
    print(y.size())

    x = torch.randn(4,256,28,28,8) # 8 NodeOp
    net = NodeLayer(in_channels=256, out_channels=256, in_edge_list=np.array([2,3]))
    net = net.to(device)
    # print(list(net.parameters()))
    x = x.to(device)
    y = net(x)
    print(y.size())

    randstate = np.random.RandomState(1)
    
    print("DAGLayer")
    x = torch.randn(4,156,2,2) # 1 input node
    print(x.size())
    
    net = DAGLayer(in_channels=156, out_channels=312, G=nx.random_regular_graph(4,32,randstate))

    net = net.to(device)
    x = x.to(device)

    y = net(x)
    print(y.size())

    params = 0
    for p in net.parameters():
        # print(p)
        if p.requires_grad:
            params += p.numel()
            
    print(params)  # 121898

    x = torch.randn(256,78,4,4) # 1 input node (B,C,P,P)
    print(x.size())
    
    net = DAGLayer(in_channels=78, out_channels=156, G=nx.random_regular_graph(4,32,randstate))

    net = net.to(device)
    x = x.to(device)

    y = net(x)
    print(y.size())

    params = 0
    for p in net.parameters():
        # print(p)
        if p.requires_grad:
            params += p.numel()
            
    print(params)  # 121898

    print("RWNN")
    Gs = [nx.random_regular_graph(4,32,randstate) for _ in range(3)]
    x = torch.randn(1,3,224,224) # 1 input node
    print(x.size())
    
    net = RWNN(78,1000,3,Gs)

    x = x.to(device)
    net = net.to(device)

    y = net(x)
    print(y.size())

    params = 0
    for p in net.parameters():
        # print(p)
        if p.requires_grad:
            params += p.numel()
            
    print(params)  # 121898
    
    # from torchinfo import summary
    # from torchvision.models import resnet18

    # # model = resnet18()
    # model = net
    # batch_size = 2

    # summary(
    #     model,
    #     input_size=(batch_size, 256, 224, 224),
    #     col_names=["output_size", "num_params"],
    # )

    # # from torchsummary import summary

    # # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # net.to(device)

    # # summary(net, (3, 32, 32))
