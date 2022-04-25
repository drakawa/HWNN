import yaml
import sys
import os

if __name__ == "__main__":
        
    ns = ["rwnn"]
    gs = ["ws"]
    ss = list(range(1,11))
    rm_revs = [("random", False), ("bfs", False), ("dfs", False), ("bfs", True), ("dfs", True)]

    obj = {"nets": ns, "graphs": gs, "seeds": ss, "reorder_method_revs": rm_revs}
    yaml_filename = "ws_dag.yaml"

    with open(yaml_filename, "w") as f:
        yaml.safe_dump(obj, f)
