import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import matplotlib as mpl
from itertools import product, combinations, combinations_with_replacement
from singular_angles import SingularAngles
import os
import networkx as nx
import random
from scipy import stats
import xarray as xr


# set fonttype so Avenir can be used with pdf format
mpl.rcParams['pdf.fonttype'] = 42
sns.set(font='Avenir', style="ticks")


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


def directed_configuration_model(size, mean_connection_prob, np_seed=1):

    def calc_degrees(length, total_connections, np_seed):
        np.random.seed(np_seed)
        connections = np.ones(total_connections)
        degrees = np.zeros(length)
        np.add.at(degrees, np.random.choice(range(length), total_connections, replace=True), connections)
        return degrees.astype(int)

    total_connections = int(mean_connection_prob * size[0] * size[1])
    indegrees = calc_degrees(size[1], total_connections, np_seed)
    outdegrees = calc_degrees(size[0], total_connections, np_seed + 1)

    graph = nx.directed_configuration_model(in_degree_sequence=indegrees, out_degree_sequence=outdegrees)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    while graph.number_of_edges() < total_connections:
        nodes = np.random.choice(graph.nodes(), size=2, replace=False)
        graph.add_edge(nodes[0], nodes[1])

    while graph.number_of_edges() > total_connections:
        edge = np.random.choice(graph.edges())
        graph.remove_edge(*edge)

    return nx.to_numpy_array(graph)


def watts_strogatz(size, mean_connection_prob, p=0.3):
    graph = nx.watts_strogatz_graph(size[0], k=int(mean_connection_prob * (size[1] - 1)), p=p)
    return nx.to_numpy_array(graph)


def barabasi_albert(size, mean_connection_prob):
    graph = nx.barabasi_albert_graph(size[0], m=int(mean_connection_prob * (size[1] - 1) / 2))
    return nx.to_numpy_array(graph)


def plot_connectome(connectome, name, title, fig=None, ax=None, save=True, cmap='Greens'):
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
    im = ax.imshow(connectome, cmap=cmap, vmin=0, vmax=np.max(connectome))
    cbar = fig.colorbar(im, ax=ax)
    ax.set_xlabel('pre-synaptic neuron')
    ax.set_ylabel('post-synaptic neuron')
    ax.set_title(title)
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if save:
        plt.savefig(f'plots/{name}.png', dpi=600)
    return ax


size_square = (300, 300)
size_rectangular = (200, 450)
mean_connection_prob = 0.1
shuffle_ = True

mean_num_connections_square = mean_connection_prob * size_square[0] * size_square[1]
mean_num_connections_rectangular = mean_connection_prob * size_rectangular[0] * size_rectangular[1]


connectomes_square = {}
connectomes_square['ER'] = erdos_renyi_connectome(size_square, mean_connection_prob)
connectomes_square['DCM'] = directed_configuration_model(size_square, mean_connection_prob)
connectomes_square['one_cluster'] = clustered_connectome(size_square, clusters=[(0, 50)], rel_cluster_weights=[100],
                                                         mean_connection_prob=mean_connection_prob)
connectomes_square['two_clusters'] = clustered_connectome(size_square, clusters=[(50, 100), (100, 120)],
                                                          rel_cluster_weights=[50, 50],
                                                          mean_connection_prob=mean_connection_prob)
connectomes_square['WS'] = watts_strogatz(size_square, mean_connection_prob, p=0.3)
connectomes_square['BA'] = barabasi_albert(size_square, mean_connection_prob)

connectomes_rectangular = {}
connectomes_rectangular['ER'] = erdos_renyi_connectome(size_rectangular, mean_connection_prob)
connectomes_rectangular['one_cluster'] = clustered_connectome(size_rectangular, clusters=[(0, 50)],
                                                              rel_cluster_weights=[100],
                                                              mean_connection_prob=mean_connection_prob)
connectomes_rectangular['two_clusters'] = clustered_connectome(size_rectangular, clusters=[(50, 100), (100, 150)],
                                                               rel_cluster_weights=[20, 80],
                                                               mean_connection_prob=mean_connection_prob)

if shuffle_:
    connectomes_square['one_cluster_shuffled'] = shuffle(connectomes_square['one_cluster'])
    connectomes_rectangular['one_cluster_shuffled'] = shuffle(connectomes_rectangular['one_cluster'])

connectome_dict = {'square': connectomes_square, 'rectangular': connectomes_rectangular}

# ------- CALCULATE SIMILARITY SCORES ACROSS CONNECTOMES -------

singular_angles = SingularAngles()

# plot instantiated connectomes
os.makedirs('plots', exist_ok=True)
# compare all connectomes with each other
titles = {
    'ER': 'Erdős-Rényi',
    'DCM': 'Directed configuration model',
    'one_cluster': 'One cluster',
    'two_clusters': 'Two clusters',
    'one_cluster_shuffled': 'One cluster - shuffled',
    'two_clusters_shuffled': 'Two clusters - shuffled',
    'WS': 'Watts-Strogatz',
    'BA': 'Barabasi-Albert',
}

score_name = 'scores'

try:
    scores = np.load(f'{score_name}.npy', allow_pickle=True).item()
    print('Scores found on disk. Continuing...')
except FileNotFoundError:
    print('Scores not found on disk. Calculating...')
    scores = {}
    for connectome_type, connectomes in connectome_dict.items():
        scores[connectome_type] = {}
        for rule_1, rule_2 in combinations_with_replacement(connectomes.keys(), 2):
            score = singular_angles.similarity(connectomes[rule_1], connectomes[rule_2])
            scores[connectome_type][f'{rule_1}-{rule_2}'] = score
            print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(score), 2)} ± "
                  f"{np.round(np.std(score), 2)}")
    np.save(f'{score_name}.npy', scores)

# calculate p values between distributions
_, p_values = stats.ttest_ind(scores['square']['ER-ER'], scores['square']['ER-BA'], equal_var=False)

comparisons = ['ER-ER', 'DCM-DCM', 'ER-DCM', 'one_cluster-one_cluster', 'ER-one_cluster',
               'two_clusters-two_clusters', 'ER-two_clusters', 'one_cluster-two_clusters', 'WS-WS', 'BA-BA', 'ER-WS',
               'ER-BA', 'one_cluster-BA']
dummy_data_array = np.zeros((len(comparisons), len(comparisons)))
p_values = xr.DataArray(
    dummy_data_array,
    coords={'comparison_0': comparisons, 'comparison_1': comparisons},
    dims=['comparison_0', 'comparison_1'])

for meta_comparisons in combinations(comparisons, 2):
    _, p_value = stats.ttest_ind(
        scores['square'][meta_comparisons[0]], scores['square'][meta_comparisons[1]], equal_var=False)
    p_values.loc[meta_comparisons[0], meta_comparisons[1]] = p_value

p_values = np.asarray(p_values)
# set values of true 0 to np.nan such that they are not plotted later on
p_values[np.isclose(p_values, 0, atol=1e-300)] = np.nan


# ------- PLOT SIMILARITY SCORES AND P VALUES FOR CONNECTOMES -------


def colormap(base_color):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_colormap', ['#FFFFFF', base_color])
    return cmap


def plot_legend(ax, hs, ls):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(hs, ls, frameon=False, loc=(0.1, 0.1), ncol=1)


colors = {
    'ER': '#332288',
    'DCM': '#88CCEE',
    'one_cluster': '#44AA99',
    'one_cluster_shuffled': '#999933',
    'two_clusters': '#CC6677',
    'WS': '#DDCC77',
    'BA': '#EE8866',
}

colors_comparisons = {
    'DCM-DCM': '#88CCEE',
    'ER-ER': '#332288',
    'one_cluster-one_cluster': '#44AA99',
    'two_clusters-two_clusters': '#CC6677',
    'one_cluster_shuffled-one_cluster_shuffled': '#5AAE61',
    'WS-WS': '#DDCC77',
    'BA-BA': '#EE8866',
    'ER-DCM': '#88CCEE',
    'ER-one_cluster': '#44AA99',
    'ER-two_clusters': '#CC6677',
    'one_cluster-two_clusters': '#000000',
    'one_cluster-one_cluster_shuffled': '#D9F0D3',
    'ER-WS': '#DDCC77',
    'ER-BA': '#EE8866',
    'one_cluster-BA': '#44AA99',
}

labels = {
    'ER-ER': 'ER - ER',
    'ER-DCM': 'DCM - ER',
    'DCM-DCM': 'DCM - DCM',
    'ER-ER': 'ER - ER',
    'ER-one_cluster': 'ER - one cluster',
    'ER-two_clusters': 'ER - two clusters',
    'one_cluster-one_cluster': 'one cluster - one cluster',
    'one_cluster-two_clusters': 'one cluster - two clusters',
    'one_cluster-one_cluster_shuffled': 'one cluster -\none cluster shuffled',
    'two_clusters-two_clusters': 'two clusters - two clusters',
    'one_cluster_shuffled-one_cluster_shuffled': 'one cluster shuffled -\none cluster shuffled',
    'ER-WS': 'ER - WS',
    'WS-WS': 'WS - WS',
    'ER-BA': 'ER - BA',
    'BA-BA': 'BA - BA',
    'one_cluster-BA': 'One cluster - BA',
}


def plot_connectome_similarity(connectomes, savename):

    mosaic = """
        AAABBB.CCCCX
        AAABBB.CCCCX
        AAABBB.CCCCX
        DDDEEE.FFFFY
        DDDEEE.FFFFY
        DDDEEE.FFFFY
        GGGHHH.IIIIZ
        GGGHHH.IIIIZ
        GGGHHH.IIIIZ
        """
    fig = plt.figure(figsize=(15, 10), layout="constrained", dpi=1200)
    ax_dict = fig.subplot_mosaic(mosaic)

    # --- PLOT CONNECTOMES ---

    ax = ax_dict['A']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['ER'], repetitions=1)[0],
                         name='square_ER', title=titles['ER'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['ER']))
    ax.text(-0.1, 1.1, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['B']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['DCM'], repetitions=1)[0],
                         name='square_DCM', title=titles['DCM'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['DCM']))
    ax.text(-0.1, 1.1, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['D']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['one_cluster'], repetitions=1)[0],
                         name='square_one_cluster', title=titles['one_cluster'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['one_cluster']))
    ax.text(-0.1, 1.1, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['E']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['two_clusters'], repetitions=1)[0],
                         name='square_two_clusters', title=titles['two_clusters'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['two_clusters']))
    ax.text(-0.1, 1.1, 'E', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['G']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['WS'], repetitions=1)[0],
                         name='square_WS', title=titles['WS'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['WS']))
    ax.text(-0.1, 1.1, 'G', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['H']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['BA'], repetitions=1)[0],
                         name='square_BA', title=titles['BA'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['BA']))
    ax.text(-0.1, 1.1, 'H', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['C']
    comparisons = ['ER-ER', 'DCM-DCM', 'ER-DCM']
    ax = singular_angles.plot_similarities(
        similarity_scores={key: scores['square'][key] for key in comparisons},
        colors=colors_comparisons, labels={c: labels[c] for c in comparisons}, ax=ax, legend=False)
    ax.text(-0.1, 1.1, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    hs, ls = ax.get_legend_handles_labels()
    plot_legend(ax_dict['X'], hs, ls)

    ax = ax_dict['F']
    comparisons = [
        'one_cluster-one_cluster', 'ER-one_cluster', 'two_clusters-two_clusters', 'ER-two_clusters',
        'one_cluster-two_clusters']#, 'one_cluster_shuffled-one_cluster_shuffled', 'one_cluster-one_cluster_shuffled']
    ax = singular_angles.plot_similarities(
        similarity_scores={key: scores['square'][key] for key in comparisons},
        colors=colors_comparisons, labels={c: labels[c] for c in comparisons}, ax=ax, legend=False)
    ax.text(-0.1, 1.1, 'F', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    hs, ls = ax.get_legend_handles_labels()
    plot_legend(ax_dict['Y'], hs, ls)

    ax = ax_dict['I']
    comparisons = [
        'WS-WS', 'BA-BA', 'ER-WS', 'ER-BA', 'one_cluster-BA']
    ax = singular_angles.plot_similarities(
        similarity_scores={key: scores['square'][key] for key in comparisons},
        colors=colors_comparisons, labels={c: labels[c] for c in comparisons}, ax=ax, legend=False)
    ax.text(-0.1, 1.1, 'I', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    hs, ls = ax.get_legend_handles_labels()
    plot_legend(ax_dict['Z'], hs, ls)

    plt.savefig(f'plots/connectomes_and_similarity_{savename}.pdf')


def plot_p_values(p_values, savename='p_values'):

    # plot p values
    # Define colormap for p-values
    top = cm.get_cmap('Blues', 1000)
    bottom = cm.get_cmap('Reds', 1000)
    newcolors = np.vstack((top(np.linspace(0.95, 0.6, 1000)),
                           bottom(np.linspace(0.45, 0.9, 1000))))
    newcmp = ListedColormap(newcolors)
    sig = 0.05 / (len(comparisons) * (len(comparisons) - 1)) / 2
    sig_alpha = np.log10(sig)
    newnorm = TwoSlopeNorm(vmin=-50, vcenter=sig_alpha, vmax=0)
    n = len(comparisons)

    mosaic = """
        AAAAAAAAAAAAAA..
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        AAAAAAAAAAAAAA.B
        """
    fig = plt.figure(figsize=(10, 9), layout="constrained")
    ax_dict = fig.subplot_mosaic(mosaic)

    # Plot color mesh
    ax = ax_dict['A']
    pval_mesh = ax.pcolormesh(np.log10(p_values.T), cmap=newcmp, norm=newnorm)
    # Add text
    for x in range(n):
        for y in range(n):
            try:
                if np.log10(p_values[x, y]) > -1:
                    ax.text(
                        x + 0.5, y + 0.5, s=f'{np.round(np.log10(p_values[x, y]), 2)}', va='center', ha='center', color='white', fontsize=10)
                else:
                    ax.text(x + 0.5, y + 0.5, s=f'{int(np.log10(p_values[x, y]))}',
                            va='center', ha='center', color='white', fontsize=10)
            except OverflowError:
                pass
            except ValueError:
                pass
    # Add white lines around each entry
    ax.set_xticks(np.arange(0, n + 1, step=0.5), minor=True)
    ax.set_yticks(np.arange(0, n + 1, step=0.5), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Format ticks
    ax.set_xticks(np.arange(0.5, n - 1))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(1.5, n))
    ax.set_yticklabels([])
    ax.set_xticklabels([labels[c] for c in comparisons][:-1], rotation=45, ha='right', va='top')
    ax.set_yticklabels([labels[c] for c in comparisons][1:])
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)

    # Inset for the colorbar
    cb = mpl.colorbar.ColorbarBase(ax_dict['B'], cmap=newcmp,
                                   norm=newnorm,
                                   boundaries=np.arange(-51, 0, step=0.1),
                                   # orientation='vertical',
                                   ticks=[-50, -40, -30, -20, -10, sig_alpha, 0])
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(6)
    cb.ax.set_xlabel('log of p-value')
    # cb.ax.set_xlim(-21, -0.1)

    plt.savefig(f'plots/{savename}.pdf')

# plot_connectome_similarity(connectomes_square, 'square')
# plot_p_values(p_values)

# plot_connectome_similarity(connectomes_rectangular, 'rectangular')

# --- CALCULATE SIMILARITY FOR INCREASINGLY DIFFERENT MATRICES


def change_matrix(base_matrix_template, max_change_fraction=0.1, step_size=0.01, repetitions=10):

    base_matrix = singular_angles.draw(base_matrix_template, repetitions=1)[0]
    size = int(np.shape(base_matrix)[0] * np.shape(base_matrix)[1])
    max_changes = int(size * max_change_fraction)

    changes = []
    similarity_mean = []
    similarity_std = []
    for change in np.arange(max_changes + 1, step=size * step_size).astype(int):
        similarity = []
        for rep in range(repetitions):
            changed_matrix = base_matrix.copy()
            indices = list(product(np.arange(np.shape(changed_matrix)[0]), np.arange(np.shape(changed_matrix)[1])))
            if change > 0:
                samples = random.sample(indices, change)
                for coordinate in samples:
                    if changed_matrix[coordinate] == 0:
                        changed_matrix[coordinate] += 1
                    else:
                        changed_matrix[coordinate] -= 1
            similarity.append(singular_angles.compare(base_matrix, changed_matrix))
        similarity_mean.append(np.mean(similarity))
        similarity_std.append(np.std(similarity))
        changes.append(change)

    similarity_mean = np.array(similarity_mean)
    similarity_std = np.array(similarity_std)
    changes = np.array(changes)

    return similarity_mean, similarity_std, changes, base_matrix


def calculate_dropoff(
        sizes=[(300, 300), (200, 450), (100, 900), (50, 1800)],
        connectome_types=['DCM', 'ER', 'one_cluster', 'two_clusters'],
        percent_evals=[0.2, 0.4, 0.6, 0.8, 1.0],
        savename='similarity_under_changes'):

    fig, axes = plt.subplots(1, len(sizes), figsize=(int(len(sizes) * 5), 5))

for i, size in enumerate(sizes):
    for connectome_type in connectome_types:
        if connectome_type == 'DCM':
            connectome = directed_configuration_model(size, mean_connection_prob)
        elif connectome_type == 'ER':
            connectome = erdos_renyi_connectome(size, mean_connection_prob)
        elif connectome_type == 'one_cluster':
            connectome = clustered_connectome(size=size, clusters=[(0, 50)], rel_cluster_weights=[100],
                                              mean_connection_prob=mean_connection_prob)
        elif connectome_type == 'two_clusters':
            connectome = clustered_connectome(size, clusters=[(0, 40), (40, 50)],
                                              rel_cluster_weights=[50, 50], mean_connection_prob=mean_connection_prob)
        similarity_mean, similarity_std, changes, base_matrix = change_matrix(connectome)
        ax = axes[i]
        x = changes / (np.shape(base_matrix)[0] * np.shape(base_matrix)[1]) * 100
        ax.plot(x, similarity_mean)
        ax.fill_between(x, similarity_mean - similarity_std, similarity_mean + similarity_std, alpha=0.2)
        ax.set_xlabel('% of changed connections')
        ax.set_ylabel('similarity')
        # ax.set_ylim(0, 1)
        ax.set_xscale('log')
            ax.set_yscale('log')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.savefig(f'plots/{savename}.png', dpi=600, bbox_inches='tight')

        # similarity_matrix = np.empty((len(sizes), len(percent_evals)))
        # for i, size in enumerate(sizes):
    #     results = change_matrix(connectomes_square[connectome_type])
    #     similarity_matrix[i] = results[0]
    #     changes = results[2]

        # ax = axes[1]
        # ax.

# calculate_dropoff()
