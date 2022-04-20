import argparse
import networkx as nx
import numpy as np

class GConfig:
    def __init__(self, d: int = None, n: int = None, s: int = None, p: float = None, g: int = None, m: int = None, name: str = None):
        self.d: int = d
        self.n: int = n
        self.s: int = s
        self.p: float = p
        self.g: int = g
        self.m: int = m
        self.name: str = name

class GenGs:
    def __init__(self, config: GConfig):
        self.config = config
        self.num_graphs = 3

    def gen_Gs(self):
        if self.config.name == "2dtorus":
            G = nx.grid_graph([self.config.n, self.config.m], periodic=True)
            G_nodes = sorted(list(G.nodes()))
            mapping = {node: G_nodes.index(node) for node in G_nodes}

            G = nx.relabel_nodes(G, mapping)
            Gs = [G.copy() for _ in range(self.num_graphs)]
            name = f"2dtorus_{self.config.m}-{self.config.m}"

        elif self.config.name == "rrg":
            randstate = np.random.RandomState(self.config.s)
            Gs = [nx.random_regular_graph(self.config.d, self.config.n, randstate) for _ in range(self.num_graphs)]
            name = f"rrg_{self.config.d}-{self.config.n}-{self.config.s}"

        elif self.config.name == "ws":
            randstate = np.random.RandomState(self.config.s)
            Gs = [nx.connected_watts_strogatz_graph(self.config.n, self.config.d, self.config.p, seed=randstate) for _ in range(self.num_graphs)]
            name = f"ws_{self.config.n}-{self.config.d}-{self.config.p}-{self.config.s}"

        # elif self.config.

        else:
            print("Invalid config:", self.config)
            exit(1)

        return Gs, name

if __name__ == "__main__":
    g_config = GConfig(n=8,m=4,name="2dtorus")
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())

    g_config = GConfig(n=32,d=4,s=1,name="rrg")
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())

    g_config = GConfig(n=32,d=4,p=0.75,s=1,name="ws")
    gen_Gs = GenGs(g_config)
    Gs, name = gen_Gs.gen_Gs()
    print(Gs, name)
    print(Gs[0].edges())

