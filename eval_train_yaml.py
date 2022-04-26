import yaml
import sys
import argparse
import argcomplete
import itertools as it

def load_yaml(yaml_path):
    try:
        with open(yaml_path) as f:
            obj = yaml.safe_load(f)
            # print(obj)
    except Exception as e:
        print(e)
        exit(1)

    return obj
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RWNN evaluation from yaml file')
    parser.add_argument('yaml_path', type=str, help='yaml filepath')

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    # print(args)
    yaml_path = args.yaml_path

    yaml_obj = load_yaml(yaml_path)
    ns = yaml_obj.get("nets")
    gs = yaml_obj.get("graphs", [None])
    ss = yaml_obj.get("seeds", [None])
    rm_revs = yaml_obj.get("reorder_method_revs", [(None, None)])
    # ns, gs, ss, rm_revs = yaml_obj["nets"],  yaml_obj["graphs"],  yaml_obj["seeds"],  yaml_obj["reorder_method_revs"]

    # print(ns, gs, ss, rm_revs)

    for n, g, (r, is_rev), s in it.product(ns, gs, rm_revs, ss):
        args_str = ""
        if n:
            args_str += "-n {} ".format(n)
        if g:
            args_str += "-g {} ".format(g)
        if s:
            args_str += "-s {} ".format(s)
        if r:
            if is_rev:
                rev_str = " --rev"
            else:
                rev_str = ""
            args_str += "-r {}{} ".format(r, rev_str)
        print("python main_cifar10.py {}-m train".format(args_str))

    # for n, g, (r, is_rev), s in it.product(ns, gs, rm_revs, ss):
    #     if is_rev:
    #         rev_str = " --rev"
    #     else:
    #         rev_str = ""
    #     print("python main_cifar10.py -n {} -g {} -s {} -r {}{} -m train".format(n, g, s, r, rev_str))
    
# ss = list(range(1,11))
# gs = ["rrg","symsa","ws"]

# for g in gs:
#     for s in ss:
#         print("python main_cifar10.py -n rwnn -g {} -s {} -m train".format(g, s))
