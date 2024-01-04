import numpy as np
import argparse
from pathlib import Path
import sys
import inspect
sys.path.append(str(Path(inspect.getfile(lambda: None)).parents[1]))
sys.path.append(str(Path(inspect.getfile(lambda: None)).parents[2]))
import config
from connectomes import Connectomes


def generate_matrix(connectome_type, N, M):
    size = (N,M)
    if N == M:
        config.network_params['size']['square'] = size
    else:
        config.network_params['size']['rectangular'] = size

    C = Connectomes(config.network_params)

    if connectome_type == 'DCM':
        connectome = C.directed_configuration_model(size)
    elif connectome_type == 'ER':
        connectome = C.erdos_renyi(size)
    elif connectome_type == '1C':
        connectome = C.clustered_connectome(size=size, 
                                            clusters=C.params['1C']['clusters'],
                                            rel_cluster_weights=C.params['1C']['rel_cluster_weights'])
    elif connectome_type == '2C':
        connectome = C.clustered_connectome(size=size, 
                                            clusters=C.params['2C']['clusters'],
                                            rel_cluster_weights=C.params['2C']['rel_cluster_weights'])
    elif connectome_type == 'WS':
        connectome = C.watts_strogatz(size, C.params['WS']['p'])
    elif connectome_type == 'BA':
        connectome = C.barabasi_albert(size)

    return connectome


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output", nargs='?', type=Path)
    CLI.add_argument("--model", nargs='?', type=str)
    CLI.add_argument("--N", nargs='?', type=int)
    CLI.add_argument("--M", nargs='?', type=int)
    args, unknown = CLI.parse_known_args()

    matrix = generate_matrix(args.model, 
                             N=args.N, 
                             M=args.M)

    np.save(args.output, matrix)
