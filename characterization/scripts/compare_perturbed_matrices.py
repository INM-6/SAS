import numpy as np
from copy import copy
from pathlib import Path
from itertools import product
import random
import inspect
import sys
sys.path.append(str(Path(inspect.getfile(lambda: None)).parents[1]))
sys.path.append(str(Path(inspect.getfile(lambda: None)).parents[2]))
import config 
from singular_angles import SingularAngles
import pandas as pd
import argparse

def evaluate_matrix_perturbation(base_matrix, max_change_fraction=0.1, 
                                 step_size=0.01, repetitions=5):
    size = int(np.shape(base_matrix)[0] * np.shape(base_matrix)[1])
    max_changes = int(size * max_change_fraction)

    changes = np.insert(np.arange(max_changes+1, step=size*step_size).astype(int),
                        1, np.arange(1, 100, step=10))
    
    df = pd.DataFrame()
    for _ in range(repetitions):
        for num_changes in changes:
            score = compare_perturbed_matrix(base_matrix, num_changes)
            d = pd.DataFrame.from_dict(dict(num_changes=[num_changes], 
                                            score=[score]))
            df = pd.concat([df, d], ignore_index=True)
    return df

def compare_perturbed_matrix(base_matrix, num_changes):
    changed_matrix = perturb_matrix(base_matrix=base_matrix, 
                                    num_changes=num_changes)
    singular_angles = SingularAngles()
    score = singular_angles.compare(base_matrix, changed_matrix)
    return score


def perturb_matrix(base_matrix, num_changes):
    changed_matrix = base_matrix.copy()
    N, M = changed_matrix.shape
    indices = list(product(np.arange(N), np.arange(M)))
    
    if num_changes > 0:
        samples = random.sample(indices, num_changes)
        for coordinate in samples:
            if changed_matrix[coordinate] == 0:
                changed_matrix[coordinate] += 1
            else:
                changed_matrix[coordinate] -= 1
    return changed_matrix


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--input", nargs='+', type=str, required=True)
    CLI.add_argument("--output", nargs='?', type=Path, required=True)
    CLI.add_argument("--model_names", nargs='+', type=str)
    CLI.add_argument("--max_change_fraction", nargs='?', type=float)
    CLI.add_argument("--step_size", nargs='?', type=float)
    CLI.add_argument("--repetitions", nargs='?', type=int)
    CLI.add_argument("--N", nargs='?', type=int)
    CLI.add_argument("--M", nargs='?', type=int)
    args, unknown = CLI.parse_known_args()

    df = pd.DataFrame()
    for model_name, file_name in zip(args.model_names, args.input):
        base_matrix = np.load(file_name)
        
        N, M = base_matrix.shape
        assert(N==N)
        assert(M==M)

        d = evaluate_matrix_perturbation(base_matrix=base_matrix,
                                    max_change_fraction=args.max_change_fraction,
                                    step_size=args.step_size,
                                    repetitions=args.repetitions)
        d['model'] = model_name
        d['N'], d['M'] = N, M
        df = pd.concat([df, d], ignore_index=True)

    args.output.parent.mkdir(exist_ok=True, parents=True)

    # if args.output.exists():
    #     old_df = pd.read_csv(args.output)
    #     df = pd.concat([old_df, df], ignore_index=True)
        
    df.to_csv(args.output)
