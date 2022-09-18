# Requirement
- pytorch
- networkx
- numpy
- scipy
- pandas
- matplotlib

# Conventional Randomly Wired Neural Network
- train for 100 epochs
```
$ python main_cifar10.py -n rwnn -g ws -m train
```

- test for epoch #100
```
$ python main_cifar_10.py -n rwnn -m test
```

# Optiamlly Wired Neural Network (point-symmetrical graph)
- train for 100 epochs
```
$ python main_cifar10.py -n rwnn -g symsa -m train
```
- test for epoch #100
```
$ python main_cifar10.py -n rwnn -g symsa -m test
```

