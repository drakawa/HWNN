import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import networkx as nx
import sys
import matplotlib.pyplot as plt
import numpy as np

from net import RWNN, ResNet50
from gen_graph import GConfig, GenGs
import argparse

import os

class DataLoader:
    def __init__(self):
        self.batch_size = 256
    
    def train_loader(self):
        batch_size = self.batch_size
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
    
        return trainloader
    
    def test_loader(self):
        batch_size = self.batch_size
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
    
        return testloader

class EvalNet:
    def __init__(self):
        self.net: nn.Module = None
        self.criterion: nn.Module = None
        self.optimizer: nn.Module = None
        self.scheduler: nn.Module = None
        self.net_name: str = None
        self.num_epochs: int = None

    def train(self):

        pths_path = "./pths/{}/".format(self.net_name)
        if not os.path.isdir(pths_path):
            os.makedirs(pths_path)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        net = self.net
        criterion = self.criterion.to(device)
        optimizer = self.optimizer
        scheduler = self.scheduler
        num_epochs = self.num_epochs
        
        dataloader = DataLoader()
        trainloader = dataloader.train_loader()

        net = net.to(device)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
        
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                # if i % 2000 == 1999:    # print every 2000 mini-batches
                #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                #     running_loss = 0.0
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.10f}')
                running_loss = 0.0
        
            PATH = "./pths/{}/cifar_net_{:0=3}.pth".format(self.net_name, epoch+1)
            torch.save(net.state_dict(), PATH)
        
            scheduler.step()
            
        print('Finished Training')
        
    def test(self, num_chkpt):
        net = self.net
        
        dataloader = DataLoader()
        testloader = dataloader.test_loader()
                
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print(device)
        
        net = net.to(device)
        PATH = "./pths/{}/cifar_net_{:0=3}.pth".format(self.net_name, num_chkpt)
        net.load_state_dict(torch.load(PATH))

        net.eval()
        
        criterion = self.criterion.to(device)

        correct = 0
        total = 0
        test_loss = 0.0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = net(images)

                test_loss += criterion(outputs, labels).item()
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"epoch {num_chkpt}: ")
        print(f"Average test loss: {test_loss / total:.8f}")
        print(f'Accuracy of the network on the 10000 test images: {correct / total:.8f}')

    def param(self):
        params = 0
        for p in self.net.parameters():
            if p.requires_grad:
                params += p.numel()
                
        print(params)  # 121898

class EvalRWNN(EvalNet):
    def __init__(self, g_config: GConfig):
        super().__init__()

        gen_Gs = GenGs(g_config)
        Gs, name = gen_Gs.gen_Gs()
        # seed = 1
        # randstate = np.random.RandomState(seed)
        # Gs = [nx.random_regular_graph(4,32,randstate) for _ in range(3)]
        
        self.net = RWNN(78,10,3,Gs)
        self.num_epochs = 100

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        self.net_name = "rwnn_%s" % name
        
class EvalResNet50(EvalNet):
    def __init__(self):
        super().__init__()

        self.net = ResNet50(num_classes=10)
        self.num_epochs = 100

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        self.net_name = "resnet50"
        
if __name__ == "__main__":

    nets = ["rwnn", "resnet50"]
    graphs_rwnn = ["rrg", "ws", "symsa", "2dtorus"]
    modes = ["train", "test", "param"]
    methods = ["random", "bfs", "dfs"]

    parser = argparse.ArgumentParser(description='NN for CIFAR-10')
    parser.add_argument('-n', '--net', type=str, help='Net name', choices=nets)
    parser.add_argument('-g', '--graph_rwnn', type=str, help='Graph name used in RWNN', choices=graphs_rwnn, required=False)
    parser.add_argument('-s', '--seed', type=int, help='Random seed used in RWNN', required=False)
    parser.add_argument('-r', '--reorder_method', type=str, help='Mode for reorder labels', choices=methods, required=False)
    parser.add_argument('--rev', action='store_true', help='labels reversed')
    parser.add_argument('-m', '--mode', type=str, help='Mode for net', choices=modes)
    parser.add_argument('-t', '--test_chkpt', default=1, type=int, help='chkpt index for test', required=False)

    args = parser.parse_args()
    print(args)
    net, graph_rwnn, seed, mode, test_chkpt, reorder_method, rev = args.net, args.graph_rwnn, args.seed, args.mode, args.test_chkpt, args.reorder_method, args.rev

    # net = sys.argv[1]
    # mode = sys.argv[2]

    if net == "rwnn":
        if graph_rwnn == "rrg":
            g_config = GConfig(n=32,d=4,s=seed,name="rrg")
        elif graph_rwnn == "ws":
            g_config = GConfig(n=32,d=4,p=0.75,s=seed,name="ws")
        elif graph_rwnn == "symsa":
            g_config = GConfig(n=32,d=4,g=4,s=seed,name="symsa")
        elif graph_rwnn == "2dtorus":
            g_config = GConfig(n=8,m=4,name="2dtorus")
        else:
            print("somethings wrong in rwnn config")
            exit(1)

        if reorder_method:
            g_config.method = reorder_method

        if rev:
            g_config.rev = rev

        eval_net = EvalRWNN(g_config)
    elif net == "resnet50":
        eval_net = EvalResNet50()
        
    if mode == "train":
        eval_net.train()
    elif mode == "test":
        # test_chkpt = int(sys.argv[3])
        eval_net.test(test_chkpt)
    elif mode == "param":
        eval_net.param()
            
