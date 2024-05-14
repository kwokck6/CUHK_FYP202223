import argparse

def get_argparse():
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group(title="general")
    general.add_argument("dataset", type=str, default=None)
    general.add_argument("model", type=str, default=None)
    general.add_argument("batch_size", type=int, default=32)
    general.add_argument("--seed", type=int, default=0)
    
    model = parser.add_argument_group(title="model")
    model.add_argument("-e", type=int, default=10)
    model.add_argument("--lr", type=float, default=1e-4)
    model.add_argument("--decay", type=float, default=0.1)

    return parser

def get_config_from_args(args):
    config = {"general": dict(),
              "model": dict()}

    config["general"]["dataset"] = args.dataset

args = get_argparse().parse_args()
config = get_config_from_args(args)
