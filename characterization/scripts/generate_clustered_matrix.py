import numpy as np
import argparse
from pathlib import Path
import inspect
import sys
sys.path.append(str(Path(inspect.getfile(lambda: None)).parents[1]))
sys.path.append(str(Path(inspect.getfile(lambda: None)).parents[2]))
from connectomes import Connectomes
import config


def generated_clustered_matrix(N, M, cluster_size, cluster_mean):
    size = (N,M)
    if N == M:
        config.network_params['size']['square'] = size
    else:
        config.network_params['size']['rectangular'] = size
    
    if cluster_size < 1: # interpret as percentage
        cluster_size = cluster_size * N

    C = Connectomes(config.network_params)
    matrix = C.clustered_connectome(size=size, 
                                    clusters=[(0,int(cluster_size))], 
                                    rel_cluster_weights=[cluster_mean])
    return matrix


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output", nargs='?', type=str)
    CLI.add_argument("--N", nargs='?', type=int)
    CLI.add_argument("--M", nargs='?', type=int)
    CLI.add_argument("--cluster_size", nargs='?', type=float)
    CLI.add_argument("--cluster_mean", nargs='?', type=float)
    CLI.add_argument("--mean_connection_prob", nargs='?', type=float)
    CLI.add_argument("--seed", nargs='?', type=int)
    args, unknown = CLI.parse_known_args()

    np.random.seed(args.seed)
    
    M = generated_clustered_matrix(N=args.N,
                                   M=args.M,
                                   cluster_size=args.cluster_size,
                                   cluster_mean=args.cluster_mean,
                                   )
    np.save(args.output, M)
