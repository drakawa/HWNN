# python main_cifar10.py -n rwnn -g ws -s 1 -m train
ss = list(range(1,11))
gs = ["rrg","symsa","ws"]

for g in gs:
    for s in ss:
        print("python main_cifar10.py -n rwnn -g {} -s {} -m train".format(g, s))

print("python main_cifar10.py -n rwnn -g {} -s {} -m train".format(""))
