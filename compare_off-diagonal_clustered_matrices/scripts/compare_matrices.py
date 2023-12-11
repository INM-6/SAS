import numpy as np
from copy import copy
from pathlib import Path
import inspect
import sys
sys.path.append(str(Path(inspect.getfile(lambda: None)).parents[1]))
from singular_angles import SingularAngles
import pandas as pd
import argparse


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output", nargs='?', type=Path, required=True)
    CLI.add_argument("--matrix_a", nargs='?', type=Path, required=True)
    CLI.add_argument("--matrix_b", nargs='?', type=Path, required=True)
    CLI.add_argument("--N", nargs='?', type=int)
    CLI.add_argument("--bkgr_mean", nargs='?', type=float)
    CLI.add_argument("--bkgr_var", nargs='?', type=float)
    CLI.add_argument("--hub_size", nargs='?', type=int)
    CLI.add_argument("--hub_mean", nargs='?', type=float)
    CLI.add_argument("--hub_var", nargs='?', type=float)
    CLI.add_argument("--seed_a", nargs='?', type=float)
    CLI.add_argument("--seed_b", nargs='?', type=float)
    args, unknown = CLI.parse_known_args()

    matrix_a = np.nan_to_num(np.load(args.matrix_a))
    matrix_b = np.nan_to_num(np.load(args.matrix_b))

    singular_angles = SingularAngles()

    score = singular_angles.compare(matrix_a, matrix_b)

    params = copy(vars(args))

    for k in ['output', 'matrix_a', 'matrix_b']:
        params.pop(k, None)
    params['score'] = score

    df = pd.Series(params).to_frame().T

    args.output.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(args.output)
