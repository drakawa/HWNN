import numpy as np

import subprocess
import pickle
import itertools as it
import os
import pandas as pd

from collections import defaultdict
from collections import defaultdict as dd

import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker

os.environ['MKL_THREADING_LAYER'] = 'GNU'

def dd_to_dict(d):
    if isinstance(d, dd):
        d = {k: dd_to_dict(v) for k, v in d.items()}
    return d

rec_dd = lambda: dd(rec_dd)

# main_cifar10.py [-h] [-n {rwnn,resnet50}] [-g {rrg,ws,symsa,2dtorus}] [-s SEED] [-m {train,test}] [-t TEST_CHKPT]
def get_loss_acc_rwnn(g, s, chkpt_idx=100):
    pickle_path = os.path.join("./savefile/loss_acc", "loss_acc_rwnn_%s_%d_%d.pickle" % (g, s, chkpt_idx))

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            avg_test_loss, top1_accuracy = pickle.load(f)
    else:
        script = "python main_cifar10.py -n rwnn -g %s -s %d -m test -t %d" % (g, s, chkpt_idx)
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

        with open(pickle_path, "wb") as f:
            pickle.dump((avg_test_loss, top1_accuracy), f)

    return avg_test_loss, top1_accuracy

def get_loss_acc_resnet50(chkpt_idx=100):
    pickle_path = os.path.join("./savefile/loss_acc", "loss_acc_resnet50_%d.pickle" % chkpt_idx)

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            avg_test_loss, top1_accuracy = pickle.load(f)
    else:
        script = "python main_cifar10.py -n resnet50 -m test -t %d" % (chkpt_idx)
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

        with open(pickle_path, "wb") as f:
            pickle.dump((avg_test_loss, top1_accuracy), f)

    return avg_test_loss, top1_accuracy

def accum_results():
    rwnn_graphs = ["ws", "rrg", "symsa"]
    chkpts = list(range(1,101))
    ids = list(range(1,11))
    # chkpts = [1,100]
    # ids = list(range(1,3))

    results = dict()
    for g, i, c in it.product(rwnn_graphs, ids, chkpts):
        results[(g,i,c)] = get_loss_acc_rwnn(g,i,c)

    for c in chkpts:
        results[("2dtorus",1,c)] = get_loss_acc_rwnn("2dtorus",1,c)
        results[("resnet50",1,c)] = get_loss_acc_resnet50(c)

    results_rec_dd = rec_dd()
    for (graph, id, chkpt), (loss, acc) in results.items():
        # print(graph, id, chkpt, loss, acc)
        results_rec_dd[graph]["loss"][chkpt][id] = loss
        results_rec_dd[graph]["acc"][chkpt][id] = acc

    results_dd = dd_to_dict(results_rec_dd)

    return results_dd

def get_plots(results_dd):
    plots_rec_dd = rec_dd()

    # label_dict = {"2dtorus": "RWNN-2d_torus", "ws": "RWNN-ws", "rrg": "RWNN-rrg", "resnet50": "ResNet-50", "symsa": "RWNN-symsa (OWNN)"}
    # label_dict = {"2dtorus": "RWNN-2d_torus", "ws": "RWNN", "rrg": "RWNN-rrg", "resnet50": "ResNet-50", "symsa": "OWNN (proposed)"}
    label_dict = {"ws": "RWNN", "resnet50": "ResNet-50", "symsa": "OWNN (proposed)"}
    graphs = ["resnet50", "ws", "symsa"]
    # graphs = ["resnet50", "ws", "symsa"]
    # graphs = ["ws", "symsa"]

    for val in ["acc", "loss"]:
        for graph in graphs:
            print(val, graph)
            df = pd.DataFrame(results_dd[graph][val]).loc[:, :]
            # print(df)
            # print(df.describe())
            # print(df.describe().loc["mean"])
            # print(df.columns)
            # print(df.index)
            plots_rec_dd[val][graph]["x"] = list(df.columns)
            plots_rec_dd[val][graph]["y"] = df.describe().loc["mean"].values.tolist()
            plots_rec_dd[val][graph]["yerr"] = df.describe().loc["std"].values.tolist()
            plots_rec_dd[val][graph]["label"] = label_dict[graph]
            # print(plots_rec_dd[val][graph])
            # print(df.describe().loc["mean"])
            # print(df.describe().loc["std"])

    plots_dd = dd_to_dict(plots_rec_dd)

    # plt.style.use(['science', "ieee"])

    plt.rcParams["font.size"] = 48
    plt.rcParams["figure.figsize"] = [36.0, 9.0]
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["lines.markeredgewidth"] = 1.0
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.framealpha"] = 1.0
    plt.rcParams["legend.labelspacing"] = 0.05

    for tick in ["xtick", "ytick"]:
        plt.rcParams[tick + ".direction"] = "in"
        plt.rcParams[tick + ".major.width"] = 1.0
        plt.rcParams[tick + ".major.size"] = 5.0
    plt.rcParams["savefig.bbox"] = "tight"
    # plt.rcParams["legend.loc"] = "bottom right"

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.rcParams["font.family"] = ["Times New Roman"]

    if False:
        print(plt.rcParams.keys())

    fmts_orig = ["-v", "->", "-^", ":s", "--o"]

    cmap = plt.get_cmap("Dark2")


    for val in ["acc", "loss"]:
        fmts = it.cycle(fmts_orig)

        plt.cla()
        fig, ax = plt.subplots()

        for p_idx, graph in enumerate(graphs):
            values = plots_dd[val][graph]
            # print(values)
            ax.errorbar(values["x"], values["y"], yerr = values["yerr"], color=cmap(p_idx), capsize=10, fmt=next(fmts), label=values["label"], markersize=10)
        ax.plot()
        ax.legend()
        ax.grid()

        ax.set_xlabel("Epoch")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.3f}".format(float(x))))

        if val == "acc":
            ax.set_ylim([0.0,1.05])
            ax.set_ylabel("Top-1 Accuracy")
        elif val == "loss":
            ax.set_ylim([-0.0005,0.0100])
            ax.set_ylabel("Average Test Loss")



    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.2f}".format(float(x))))

    # ax.set_xscale("squareroot")
    # ax.set_xticks(np.arange(0,81,20)**2, position=(0.0, -0.03))
    # ax.set_xlabel("# of switches")
    # ax.set_ylabel("Elapsed time [s]")
    # ax.set_ylim([0.0,70])

        plt.savefig("fig/loss_acc/%s_fit.eps" % val, transparent=False)
        plt.savefig("fig/loss_acc/%s_fit.png" % val, transparent=False)
        plt.savefig("fig/loss_acc/%s_fit.svg" % val, transparent=True)

    return plots_dd

if __name__ == "__main__":
    # print(get_loss_acc_rwnn("ws", 1))
    # print(get_loss_acc_resnet50())

    # print(get_loss_acc_rwnn("ws", 1, 100))
    # print(get_loss_acc_resnet50(100))

    # print(get_loss_acc_rwnn("ws", 1, 10))
    # print(get_loss_acc_resnet50(10))

    accum_results = accum_results()
    # print(accum_results)

    plots_dd = get_plots(accum_results)
    # print(plots_dd)

    resnet50_acc = pd.DataFrame(accum_results["resnet50"]["acc"]).loc[:,:].describe().loc["mean"]
    rrg_acc = pd.DataFrame(accum_results["rrg"]["acc"]).loc[:,:].describe().loc["mean"]
    ws_acc = pd.DataFrame(accum_results["ws"]["acc"]).loc[:,:].describe().loc["mean"]
    symsa_acc = pd.DataFrame(accum_results["symsa"]["acc"]).loc[:,:].describe().loc["mean"]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(resnet50_acc)
    print(rrg_acc)

    rrg_resnet_rate = rrg_acc / resnet50_acc

    print(rrg_resnet_rate[50:])

    rrg_ws_rate = rrg_acc / ws_acc
    print(rrg_ws_rate)

    print(rrg_acc)

    symsa_ws_rate = symsa_acc / ws_acc
    print(symsa_ws_rate)
    print(symsa_ws_rate.max())
    print(symsa_ws_rate.idxmax())
    print(symsa_ws_rate.nlargest(5))

    ws_acc_std = pd.DataFrame(accum_results["ws"]["acc"]).loc[:,:].describe().loc["std"]
    symsa_acc_std = pd.DataFrame(accum_results["symsa"]["acc"]).loc[:,:].describe().loc["std"]

    print(ws_acc_std)
    print(symsa_acc_std)

    ws_loss_std = pd.DataFrame(accum_results["ws"]["loss"]).loc[:,:].describe().loc["std"]
    symsa_loss_std = pd.DataFrame(accum_results["symsa"]["loss"]).loc[:,:].describe().loc["std"]

    print(ws_loss_std)
    print(symsa_loss_std)

    symsa_resnet50_acc_rate = symsa_acc / resnet50_acc
    print(1-symsa_resnet50_acc_rate[100])
