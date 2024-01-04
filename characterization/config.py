import itertools

# OFF-DIAGONAL CLUSTERED MATRIX

N = [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800]
default_N = 300
default_M = int(default_N*1.5)

cluster_size = [5,10,15,20,25,30,35,40,45,50]
default_cluster_size = 0.166 #50/300 # corresponds to 16.666% of N

cluster_mean = [0,5,10,15]
default_cluster_mean = 10

seed = [0,1,2,3,4,5,6,7,8,9,10]
seed_pairs = [f'{i[0]}-{i[1]}' for i in itertools.combinations(seed,2)]


# GRAPH MODELS
network_params = {
    'mean_connection_prob': 0.1,
    'size': {
            'square': (default_N, default_N),
            'rectangular': (default_N, default_M)},
    'WS': {'p': 0.3},
    '1C': {'clusters': [(0, 50)], 
           'rel_cluster_weights': [default_cluster_mean]},
    '2C': {'clusters': [(0, 35), (35, 50)], 
           'rel_cluster_weights': [default_cluster_mean, default_cluster_mean]},
                }

# perturbations
max_change_fraction = 0.1
step_size = 0.005
repetitions = 55


model_names = ['ER', 'DCM', '1C', '2C', 'WS', 'BA']

titles = {
    'ER': 'Erdős-Rényi',
    'DCM': 'Directed configuration model',
    '1C': 'One cluster',
    '2C': 'Two clusters',
    '1Cs': 'One cluster - shuffled',
    '2Cs': 'Two clusters - shuffled',
    'WS': 'Watts-Strogatz',
    'BA': 'Barabasi-Albert',
        }

colors = {
    'ER': '#6151AC',
    'DCM': '#88CCEE',
    '1C': '#44AA99',
    '1Cs': '#999933',
    '2C': '#CC6677',
    '2Cs': '#882255',
    'WS': '#DDCC77',
    'BA': '#EE8866',
}