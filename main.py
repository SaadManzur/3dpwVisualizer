import argparse

from core.tdpw_dataset import TDPWDataset

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--spath", type=str)
    parser.add_argument("--ipath", type=str)
    parser.add_argument("--seq", type=str)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    dataset = TDPWDataset(args.spath, args.ipath)
    dataset.view(args.seq)