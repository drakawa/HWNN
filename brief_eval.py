import subprocess
import pickle
import itertools as it
import os
import pandas as pd

from collections import defaultdict
from collections import defaultdict as dd

import numpy as np

# python main_cifar10.py -n rwnn -g rrg -s 1 -m test -t 40

def get_loss_acc(script):
    print("Run script:\n", script)
    
    output = subprocess.check_output(script.split())
    print(output)
    rows = output.decode('utf-8').split("\n")
    for row in rows:
        print(row)
        if "Average test loss" in row:
            avg_test_loss = float(row.rstrip().split()[-1])
        elif "Accuracy" in row:
            top1_accuracy = float(row.rstrip().split()[-1])

    print(avg_test_loss, top1_accuracy)
    return avg_test_loss, top1_accuracy

def dd_to_dict(d):
    if isinstance(d, dd):
        d = {k: dd_to_dict(v) for k, v in d.items()}
    return d

rec_dd = lambda: dd(rec_dd)

results_rec_dd = rec_dd()
for t in range(10,110,10):
    for g in ["rrg","symsa","ws"]:
        script = "python main_cifar10.py -n rwnn -g {} -s 1 -m test -t {}".format(g, t)
        loss, acc = get_loss_acc(script)
        results_rec_dd[g]["loss"][t] = loss
        results_rec_dd[g]["acc"][t] = acc

results_dd = dd_to_dict(results_rec_dd)

for metric in ["loss", "acc"]:
    print("\t".join(["rrg","symsa","ws"]))
    for t in range(10,110,10):
        for g in ["rrg","symsa","ws"]:
            print(results_dd[g][metric][t], end="")
            print("\t", end="")
        print()

        
