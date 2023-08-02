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


def clustered_connectome(total_size, clusters, rel_cluster_weights, mean_connection_prob):
    mean_num_connections = mean_connection_prob * size[0] * size[1]
    connectome = np.ones(size)
    for cluster, rel_cluster_weight in zip(clusters, rel_cluster_weights):
        connectome[cluster[0]:cluster[1], :][:, cluster[0]:cluster[1]] = rel_cluster_weight
    return connectome / (np.sum(connectome) / mean_num_connections)


def erdos_renyi_connectome(total_size, mean_connection_prob):
    return np.ones(size) * mean_connection_prob


def plot_connectome(connectome, name):
    fig, ax = plt.subplots()
    im = plt.imshow(connectome, cmap='Greens', vmin=0, vmax=np.max(connectome))
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('pre-synaptic neuron')
    ax.set_ylabel('post-synaptic neuron')
    plt.title(name)
    plt.savefig(f'plots/{name}.png', dpi=600)


size = (100, 100)
mean_connection_prob = 0.1

connectomes = {}
connectomes['ER'] = erdos_renyi_connectome(size, mean_connection_prob)
connectomes['one_cluster'] = clustered_connectome(size, clusters=[(0, 10)], rel_cluster_weights=[100],
                                                  mean_connection_prob=mean_connection_prob)
connectomes['two_clusters'] = clustered_connectome(size, clusters=[(20, 30), (40, 50)], rel_cluster_weights=[50, 50],
                                                   mean_connection_prob=mean_connection_prob)


# ------- CALCULATE SIMILARITY SCORES FOR CONNECTOMES -------

singular_angles = SingularAngles()

# plot instantiated connectomes
os.makedirs('plots', exist_ok=True)
for name, connectome in connectomes.items():
    plot_connectome(connectome=singular_angles.draw(connectome, repetitions=1)[0], name=name)

# compare all connectomes with each other
scores = {}
for rule_1, rule_2 in combinations_with_replacement(connectomes.keys(), 2):
    scores[f'{rule_1}-{rule_2}'] = singular_angles.similarity(connectomes[rule_1], connectomes[rule_2])
    print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(scores[f'{rule_1}-{rule_2}']), 2)} ± "
          f"{np.round(np.std(scores[f'{rule_1}-{rule_2}']), 2)}")

# ------- PLOT SIMILARITY SCORES FOR CONNECTOMES -------

colors = {
    'ER-ER': '#332288',
    'ER-one_cluster': '#88CCEE',
    'ER-two_clusters': '#44AA99',
    'one_cluster-one_cluster': '#117733',
    'one_cluster-two_clusters': '#CC6677',
    'two_clusters-two_clusters': '#882255',
}
labels = {
    'ER-ER': 'ER - ER',
    'ER-one_cluster': 'ER - one cluster',
    'ER-two_clusters': 'ER - two clusters',
    'one_cluster-one_cluster': 'one cluster - one cluster',
    'one_cluster-two_clusters': 'one cluster - two clusters',
    'two_clusters-two_clusters': 'two clusters - two clusters',
}

fig, ax = plt.subplots()
ax = singular_angles.plot_similarities(similarity_scores=scores, colors=colors, labels=labels, ax=ax)
plt.savefig('plots/similarity_scores.png', dpi=600)
