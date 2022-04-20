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
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.net_name = None
        self.num_epochs = None

    def train(self):
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
        print(device)
        
        PATH = "./pths/{}/cifar_net_{:0=3}.pth".format(self.net_name, test_chkpt)
        net.load_state_dict(torch.load(PATH))
        
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        
class EvalResNet50_orig:
    def __init__(self):
        pass

    def train(self):
        dataloader = DataLoader()
        trainloader = dataloader.train_loader()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        net = ResNet50(num_classes=10)
        net = net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                
        for epoch in range(100):  # loop over the dataset multiple times
        
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
        
            PATH = "./pths/resnet50/cifar_net_{:0=3}.pth".format(epoch+1)
            torch.save(net.state_dict(), PATH)
        
            scheduler.step()
            
        print('Finished Training')

    def test(self, num_chkpt):
        dataloader = DataLoader()
        testloader = dataloader.test_loader()
                
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        net = ResNet50(num_classes=10)

        PATH = "./pths/resnet50/cifar_net_{:0=3}.pth".format(num_chkpt)
        net.load_state_dict(torch.load(PATH))
        
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

class EvalRWNN(EvalNet):
    def __init__(self):
        super().__init__()

        seed = 1
        randstate = np.random.RandomState(seed)
        Gs = [nx.random_regular_graph(4,32,randstate) for _ in range(3)]
        
        self.net = RWNN(78,10,3,Gs)
        self.num_epochs = 100

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        self.net_name = "rwnn"
        
class EvalResNet50(EvalNet):
    def __init__(self):
        super().__init__()

        self.net = ResNet50(num_classes=10)
        self.num_epochs = 100

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        self.net_name = "resnet50"
        
class EvalRWNN_orig:
    def __init__(self):
        pass
    
    def train(self):
        dataloader = DataLoader()
        trainloader = dataloader.train_loader()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
    
        seed = 1
        randstate = np.random.RandomState(seed)
        Gs = [nx.random_regular_graph(4,32,randstate) for _ in range(3)]
        net = RWNN(78,10,3,Gs)
        
        net = net.to(device)
    
        num_epochs = 100
    
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # criterion = F.nll_loss
        # optimizer = optim.Adam(net.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
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
        
            PATH = "./pths/rwnn/cifar_net_{:0=3}.pth".format(epoch+1)
            torch.save(net.state_dict(), PATH)
        
            scheduler.step()
            
        print('Finished Training')

    def test(self, num_chkpt):
        dataloader = DataLoader()
        testloader = dataloader.test_loader()
                
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        seed = 1
        randstate = np.random.RandomState(seed)
        Gs = [nx.random_regular_graph(4,32,randstate) for _ in range(3)]
        
        net = RWNN(78,10,3,Gs)
        
        PATH = "./pths/rwnn/cifar_net_{:0=3}.pth".format(test_chkpt)
        net.load_state_dict(torch.load(PATH))
        
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        
if __name__ == "__main__":
    net = sys.argv[1]
    mode = sys.argv[2]

    if net == "rwnn":
        eval_net = EvalRWNN()
    elif net == "resnet50":
        eval_net = EvalResNet50()
        
    if mode == "train":
        eval_net.train()
    elif mode == "test":
        test_chkpt = int(sys.argv[3])
        eval_net.test(test_chkpt)
            
