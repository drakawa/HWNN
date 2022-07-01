import yaml
import sys
import os

if __name__ == "__main__":
        
    ns = ["rwnn"]
    # gs = ["ws"]
    gs = ["symsa"]
    ss = list(range(1,11))
    # rm_revs = [(None, None), ("random", False), ("bfs", False), ("dfs", False), ("bfs", True), ("dfs", True)]
    rm_revs = [("random", False), ("bfs", False)]

    obj = {"nets": ns, "graphs": gs, "seeds": ss, "reorder_method_revs": rm_revs}
    yaml_filename = "./yamls/symsa_dag2.yaml"

    with open(yaml_filename, "w") as f:
        yaml.safe_dump(obj, f)
