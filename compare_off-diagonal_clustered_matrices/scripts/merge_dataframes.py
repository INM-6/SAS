import argparse
import pandas as pd
import numpy as np
from pathlib import Path

int_types = ['N', 'hub_size', 'seed_a', 'seed_b']
float_types = ['bkgr_mean', 'bkgr_var', 'hub_mean', 'hub_var', 'score']
dtypes = dict.fromkeys(int_types, 'Int64') | dict.fromkeys(float_types, float)

def load_df(path):
    df = pd.read_csv(path, dtype=dtypes)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)],
            axis=1, inplace=True)
    return df


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--input", nargs='?', type=lambda s: s.split(' '))
    CLI.add_argument("--output", nargs='?', type=Path)
    args, unknown = CLI.parse_known_args()

    dfs = []
    for path in args.input:
        dfs.append(load_df(path))

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(args.output)
