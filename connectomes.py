import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from itertools import combinations_with_replacement
from singular_angles import SingularAngles
import os


# set fonttype so Avenir can be used with pdf format
mpl.rcParams['pdf.fonttype'] = 42
sns.set(font='Avenir')


def clustered_connectome(size, clusters, rel_cluster_weights, mean_connection_prob):
    mean_num_connections = mean_connection_prob * size[0] * size[1]
    connectome = np.ones(size)
    for cluster, rel_cluster_weight in zip(clusters, rel_cluster_weights):
        connectome[cluster[0]:cluster[1], :][:, cluster[0]:cluster[1]] = rel_cluster_weight
    return connectome / (np.sum(connectome) / mean_num_connections)


def shuffle(matrix, np_seed=1):
    np.random.seed(np_seed)
    n, m = matrix.shape
    flat_matrix = matrix.flatten()
    np.random.shuffle(flat_matrix)
    return flat_matrix.reshape(n, m)


def erdos_renyi_connectome(size, mean_connection_prob):
    return np.ones(size) * mean_connection_prob


def plot_connectome(connectome, name):
    fig, ax = plt.subplots()
    im = plt.imshow(connectome, cmap='Greens', vmin=0, vmax=np.max(connectome))
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('pre-synaptic neuron')
    ax.set_ylabel('post-synaptic neuron')
    plt.title(name)
    plt.savefig(f'plots/{name}.png', dpi=600)


size_square = (300, 300)
size_rectangular = (200, 450)
mean_connection_prob = 0.1
shuffle_ = True

mean_num_connections_square = mean_connection_prob * size_square[0] * size_square[1]
mean_num_connections_rectangular = mean_connection_prob * size_rectangular[0] * size_rectangular[1]


connectomes_square = {}
connectomes_square['ER'] = erdos_renyi_connectome(size_square, mean_connection_prob)
connectomes_square['one_cluster'] = clustered_connectome(size_square, clusters=[(0, 10)], rel_cluster_weights=[100],
                                                         mean_connection_prob=mean_connection_prob)
connectomes_square['two_clusters'] = clustered_connectome(size_square, clusters=[(20, 30), (40, 50)],
                                                          rel_cluster_weights=[20, 80],
                                                          mean_connection_prob=mean_connection_prob)

connectomes_rectangular = {}
connectomes_rectangular['ER'] = erdos_renyi_connectome(size_rectangular, mean_connection_prob)
connectomes_rectangular['one_cluster'] = clustered_connectome(size_rectangular, clusters=[(0, 10)],
                                                              rel_cluster_weights=[100],
                                                              mean_connection_prob=mean_connection_prob)
connectomes_rectangular['two_clusters'] = clustered_connectome(size_rectangular, clusters=[(20, 30), (40, 50)],
                                                               rel_cluster_weights=[20, 80],
                                                               mean_connection_prob=mean_connection_prob)

if shuffle_:
    connectomes_square['one_cluster_shuffled'] = shuffle(connectomes_square['one_cluster'])
    connectomes_rectangular['one_cluster_shuffled'] = shuffle(connectomes_rectangular['one_cluster'])

connectome_dict = {'square': connectomes_square, 'rectangular': connectomes_rectangular}

# ------- CALCULATE SIMILARITY SCORES FOR CONNECTOMES -------

singular_angles = SingularAngles()

# plot instantiated connectomes
os.makedirs('plots', exist_ok=True)
# compare all connectomes with each other
scores = {}
for connectome_type, connectomes in connectome_dict.items():
    for name, connectome in connectomes.items():
        plot_connectome(connectome=singular_angles.draw(connectome, repetitions=1)
                        [0], name=connectome_type + '_' + name)

    scores[connectome_type] = {}
    for rule_1, rule_2 in combinations_with_replacement(connectomes.keys(), 2):
        score = singular_angles.similarity(connectomes[rule_1], connectomes[rule_2])
        scores[connectome_type][f'{rule_1}-{rule_2}'] = score
        print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(score), 2)} ± "
              f"{np.round(np.std(score), 2)}")

# ------- PLOT SIMILARITY SCORES FOR CONNECTOMES -------

colors = {
    'ER-ER': '#332288',
    'ER-one_cluster': '#88CCEE',
    'ER-two_clusters': '#44AA99',
    'ER-one_cluster_shuffled': '#117733',
    'one_cluster-one_cluster': '#999933',
    'one_cluster-two_clusters': '#DDCC77',
    'one_cluster-one_cluster_shuffled': '#EE8866',
    'two_clusters-two_clusters': '#CC6677',
    'two_clusters-one_cluster_shuffled': '#882255',
    'one_cluster_shuffled-one_cluster_shuffled': '#AA4499',
}
labels = {
    'ER-ER': 'ER - ER',
    'ER-one_cluster': 'ER - one cluster',
    'ER-two_clusters': 'ER - two clusters',
    'ER-one_cluster_shuffled': 'ER - one cluster shuffled',
    'one_cluster-one_cluster': 'one cluster - one cluster',
    'one_cluster-two_clusters': 'one cluster - two clusters',
    'one_cluster-one_cluster_shuffled': 'one cluster - one cluster shuffled',
    'two_clusters-two_clusters': 'two clusters - two clusters',
    'two_clusters-one_cluster_shuffled': 'two clusters - one cluster shuffled',
    'one_cluster_shuffled-one_cluster_shuffled': 'one cluster shuffled - one cluster shuffled',
}

for connectome_type, connectomes in connectome_dict.items():
    fig, ax = plt.subplots()
    ax = singular_angles.plot_similarities(similarity_scores=scores[connectome_type], colors=colors,
                                           labels=labels, ax=ax)
    ax.set_title(connectome_type)
    plt.savefig(f'plots/similarity_scores_{connectome_type}.png', dpi=600)
