import numpy as np
import argparse


def generate_matrix(N, hub_size, hub_mean, hub_var, bkgr_mean, bkgr_var):

    M = np.random.normal(loc=bkgr_mean, 
                         scale=np.sqrt(bkgr_var), 
                         size=(N,N))
    
    hub = np.random.normal(loc=hub_mean, 
                           scale=np.sqrt(hub_var), 
                           size=(hub_size,hub_size))

    # locate the hub at the upper right (off diagonal)
    M[:hub_size, -hub_size:] = hub
    return M


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--N", nargs='?', type=int)
    CLI.add_argument("--bkgr_mean", nargs='?', type=float, default=0)
    CLI.add_argument("--bkgr_var", nargs='?', type=float, default=1)
    CLI.add_argument("--hub_size", nargs='?', type=int)
    CLI.add_argument("--hub_mean", nargs='?', type=float)
    CLI.add_argument("--hub_var", nargs='?', type=float)
    CLI.add_argument("--seed", nargs='?', type=int)
    CLI.add_argument("--output", nargs='?', type=str)
    args, unknown = CLI.parse_known_args()

    np.random.seed(args.seed)
    
    M = generate_matrix(N=args.N,
                        hub_size=args.hub_size,
                        hub_mean=args.hub_mean,
                        hub_var=args.hub_var,
                        bkgr_mean=args.bkgr_mean,
                        bkgr_var=args.bkgr_var,
                        )
    
    np.save(args.output, M)
