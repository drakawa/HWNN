import argparse
import networkx as nx
import numpy as np
import os
import pandas as pd

def reorder_G(G, method, rev=False):
    print("reorder_G:", G.nodes(), G.edges())
    center = np.random.choice(nx.center(G))
    G_nodes = sorted(list(G.nodes()))

    if method == "random":
        new_G_nodes = list(G_nodes)
        np.random.shuffle(new_G_nodes)
    elif method == "bfs":
        edges = nx.bfs_edges(G, center)
        new_G_nodes = [center] + [v for u, v in edges]
    elif method == "dfs":
        edges = nx.dfs_edges(G, center)
        new_G_nodes = [center] + [v for u, v in edges]
    else:
        print("invalid method:", method)
        exit(1)

    if method in ["bfs", "dfs"] and rev:
        new_G_nodes.reverse()

    mapping = {i:j for i,j in zip(G_nodes, new_G_nodes)}
    newG = nx.relabel_nodes(G, mapping)
    return newG

class GConfig:
    def __init__(self, d: int = None, n: int = None, s: int = 0, p: float = None, g: int = None, m: int = None, name: str = None, method: str = None, rev: bool = False):
        self.d: int = d
        self.n: int = n
        self.s: int = s
        self.p: float = p
        self.g: int = g
        self.m: int = m
        self.name: str = name
        self.method: str = method
        self.rev: bool = rev

class GenGs:
    def __init__(self, config: GConfig):
        self.config = config
        self.num_graphs = 3

    def gen_Gs(self):
        if self.config.s:
            randstate = np.random.RandomState(self.config.s)

        if self.config.name == "2dtorus":
            name = f"2dtorus_{self.config.m}_{self.config.m}"
        elif self.config.name == "rrg":
            name = f"rrg_{self.config.d}_{self.config.n}_{self.config.s}"
        elif self.config.name == "ws":
            name = f"ws_{self.config.n}_{self.config.d}_{self.config.p}_{self.config.s}"
        elif self.config.name == "symsa":
            name = f"symsa_{self.config.n}_{self.config.d}_{self.config.g}_{self.config.s}"
        else:
            print("Invalid config:", self.config)
            exit(1)

        if self.config.method == "random":
            name += "_random"
        elif self.config.method in ["bfs", "dfs"]:
            if not self.config.rev:
                name += "_" + self.config.method
            else:
                name += "_" + self.config.method + "_" + "rev"
            
        graph_path = os.path.join("./graphs", name + ".edges")
        if os.path.exists(graph_path):
            Gs = pd.read_pickle(graph_path)
        else:
            if self.config.name == "2dtorus":
                G = nx.grid_graph([self.config.n, self.config.m], periodic=True)
                G_nodes = sorted(list(G.nodes()))
                mapping = {node: G_nodes.index(node) for node in G_nodes}

                G = nx.relabel_nodes(G, mapping)
                Gs = [G.copy() for _ in range(self.num_graphs)]

            elif self.config.name == "rrg":
                Gs = [nx.random_regular_graph(self.config.d, self.config.n, randstate) for _ in range(self.num_graphs)]

            elif self.config.name == "ws":
                Gs = [nx.connected_watts_strogatz_graph(self.config.n, self.config.d, self.config.p, seed=randstate) for _ in range(self.num_graphs)]

            elif self.config.name == "symsa":
                symsa_seeds = [(self.config.s - 1) * self.num_graphs + offset + 1 for offset in range(self.num_graphs)]
                edgelists = ["symsa_edgefiles/symsa_s{}_{}_{}_{}.txt".format(self.config.g, self.config.n, self.config.d, i) for i in symsa_seeds]
                print("edgelists:", edgelists)
                Gs = [nx.read_edgelist(edgelist, nodetype=int) for edgelist in edgelists]
                # print(Gs[0].edges())

            else:
                print("Invalid config:", self.config)
                exit(1)

            if self.config.method:
                Gs = [reorder_G(G, self.config.method, self.config.rev) for G in Gs]

            pd.to_pickle(Gs, graph_path)

        return Gs, name

if __name__ == "__main__":
    g_config = GConfig(n=8,m=4,name="2dtorus")
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())

    print("rrg")
    g_config = GConfig(n=32,d=4,s=1,name="rrg")
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())
    print(Gs[1].edges())
    print(Gs[2].edges())
    print(Gs[0][0])
    print(Gs[1][0])
    print(Gs[2][0])

    g_config = GConfig(n=32,d=4,p=0.75,s=1,name="ws")
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())

    g_config = GConfig(n=32,d=4,g=4,s=3,name="symsa")
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())

    g_config = GConfig(n=32,d=4,g=4,s=3,name="symsa",method="dfs")
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())

    g_config = GConfig(n=32,d=4,g=4,s=3,name="symsa",method="random")
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())

    g_config = GConfig(n=32,d=4,g=4,s=3,name="symsa",method="bfs",rev=True)
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())

