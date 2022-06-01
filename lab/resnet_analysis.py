import networkx as nx
import itertools as it
import scipy

def to_dag(G):
    DAG = nx.DiGraph()
    DAG.add_nodes_from(G.nodes())
    for i,j in G.edges():
        i,j = sorted([i,j])
        DAG.add_edge(i,j)
    return DAG

if __name__ == "__main__":
    print("hoge")
    n_node = 8
    G = nx.complete_graph(n_node)
    DAG = to_dag(G)
    print(DAG.edges())

    in_nodes =  [n for n in DAG  if DAG.in_degree(n) == 0]
    out_nodes = [n for n in DAG if DAG.out_degree(n) == 0]

    print(in_nodes, out_nodes)

    for s, t in it.product(in_nodes, out_nodes):
        asps = nx.all_simple_paths(DAG, source=s, target=t)
        for asp in asps:
            print(asp)
        # print(list(asps))

    exit(1)

    print(nx.betweenness_centrality(DAG, endpoints=True))
    print(nx.edge_betweenness_centrality(DAG))

    Lap = nx.directed_laplacian_matrix(DAG)
    print(Lap)

    Inc = nx.incidence_matrix(DAG)
    print(Inc.todense())

    Eig = nx.eigenvector_centrality_numpy(DAG)
    print(Eig)
