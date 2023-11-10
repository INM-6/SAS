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

networks = ['ER', 'DCM', 'one_cluster', 'two_clusters', 'WS', 'BA', 'one_cluster_shuffled', 'two_clusters_shuffled']


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
    indegrees = calc_degrees(size[0], total_connections, np_seed)
    outdegrees = calc_degrees(size[1], total_connections, np_seed + 1)

    graph = nx.directed_configuration_model(in_degree_sequence=indegrees, out_degree_sequence=outdegrees)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    while graph.number_of_edges() < total_connections:
        nodes = np.random.choice(graph.nodes(), size=2, replace=False)
        graph.add_edge(nodes[0], nodes[1])

    while graph.number_of_edges() > total_connections:
        edge = np.random.choice(graph.edges())
        graph.remove_edge(*edge)

    return nx.to_numpy_array(graph).T[:size[0], :size[1]]


def watts_strogatz(size, mean_connection_prob, p=0.3):
    graph = nx.watts_strogatz_graph(np.maximum(size[0], size[1]), k=int(
        mean_connection_prob * (np.maximum(size[0], size[1]) - 1)), p=p)
    matrix = nx.to_numpy_array(graph).T
    if size[0] != size[1]:
        matrix = _rectangularize(matrix, size)
    return matrix


def barabasi_albert(size, mean_connection_prob):
    graph = nx.barabasi_albert_graph(np.maximum(size[0], size[1]), m=int(
        mean_connection_prob * (np.maximum(size[0], size[1]) - 1) / 2))
    matrix = nx.to_numpy_array(graph).T
    if size[0] != size[1]:
        matrix = _rectangularize(matrix, size)
    return matrix


def _rectangularize(matrix, size):
    random_choice = np.random.choice(np.maximum(size[0], size[1]), size=np.minimum(size[0], size[1]), replace=False)
    if size[0] < size[1]:
        matrix = matrix[random_choice, :]
    else:
        matrix = matrix[:, random_choice]
    return matrix


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
size_rectangular = (450, 200)
mean_connection_prob = 0.1
shuffle_ = True

mean_num_connections_square = mean_connection_prob * size_square[0] * size_square[1]
mean_num_connections_rectangular = mean_connection_prob * size_rectangular[0] * size_rectangular[1]


connectomes_square = {}
connectomes_square['ER'] = erdos_renyi_connectome(size_square, mean_connection_prob)
connectomes_square['DCM'] = directed_configuration_model(size_square, mean_connection_prob)
connectomes_square['one_cluster'] = clustered_connectome(size_square, clusters=[(0, 50)], rel_cluster_weights=[10],
                                                         mean_connection_prob=mean_connection_prob)
connectomes_square['two_clusters'] = clustered_connectome(size_square, clusters=[(50, 85), (85, 100)],
                                                          rel_cluster_weights=[10, 10],
                                                          mean_connection_prob=mean_connection_prob)
connectomes_square['WS'] = watts_strogatz(size_square, mean_connection_prob, p=0.3)
connectomes_square['BA'] = barabasi_albert(size_square, mean_connection_prob)

connectomes_rectangular = {}
connectomes_rectangular['ER'] = erdos_renyi_connectome(size_rectangular, mean_connection_prob)
connectomes_rectangular['DCM'] = directed_configuration_model(size_rectangular, mean_connection_prob)
connectomes_rectangular['one_cluster'] = clustered_connectome(size_rectangular, clusters=[(0, 50)],
                                                              rel_cluster_weights=[10],
                                                              mean_connection_prob=mean_connection_prob)
connectomes_rectangular['two_clusters'] = clustered_connectome(size_rectangular, clusters=[(50, 85), (85, 100)],
                                                               rel_cluster_weights=[10, 10],
                                                               mean_connection_prob=mean_connection_prob)
connectomes_rectangular['WS'] = watts_strogatz(size_rectangular, mean_connection_prob, p=0.3)
connectomes_rectangular['BA'] = barabasi_albert(size_rectangular, mean_connection_prob)

if shuffle_:
    connectomes_square['one_cluster_shuffled'] = shuffle(connectomes_square['one_cluster'])
    connectomes_square['two_clusters_shuffled'] = shuffle(connectomes_square['two_clusters'])
    connectomes_rectangular['one_cluster_shuffled'] = shuffle(connectomes_rectangular['one_cluster'])
    connectomes_rectangular['two_clusters_shuffled'] = shuffle(connectomes_rectangular['two_clusters'])

connectome_dict = {'square': connectomes_square, 'rectangular': connectomes_rectangular}

# ------- CALCULATE SIMILARITY SCORES ACROSS CONNECTOMES -------

singular_angles = SingularAngles()

# plot instantiated connectomes
os.makedirs('plots', exist_ok=True)
# compare all connectomes with each other
titles = {
    'ER': 'Erdős-Rényi',
    'DCM': 'directed configuration model',
    'one_cluster': 'one cluster',
    'two_clusters': 'two clusters',
    'one_cluster_shuffled': 'one cluster - shuffled',
    'two_clusters_shuffled': 'two clusters - shuffled',
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
    for matrix_shape, connectomes in connectome_dict.items():
        scores[matrix_shape] = {}
        for rule_1, rule_2 in combinations_with_replacement(connectomes.keys(), 2):
            score = singular_angles.similarity(connectomes[rule_1], connectomes[rule_2])
            scores[matrix_shape][f'{rule_1}-{rule_2}'] = score
            print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(score), 2)} ± "
                  f"{np.round(np.std(score), 2)}")
    np.save(f'{score_name}.npy', scores)


def calc_p_values(matrix_shape):
    # calculate p values between distributions
    comparisons = []
    for i, network_i in enumerate(networks):
        for j, network_j in enumerate(networks):
            if j >= i:
                comparisons.append(f'{network_i}-{network_j}')
    dummy_data_array = np.zeros((len(comparisons), len(comparisons)))
    p_values = xr.DataArray(
        dummy_data_array,
        coords={'comparison_0': comparisons, 'comparison_1': comparisons},
        dims=['comparison_0', 'comparison_1'])

    for meta_comparisons in combinations(comparisons, 2):

        # if self-similarity is at position 1, swap positions
        if (meta_comparisons[1].split('-')[0] == meta_comparisons[1].split('-')[1]):
            if (meta_comparisons[0].split('-')[0] == meta_comparisons[0].split('-')[1]):
                if (networks.index(meta_comparisons[1].split('-')[0])
                        < networks.index(meta_comparisons[0].split('-')[0])):
                    meta_comparisons = tuple(reversed(meta_comparisons))
            else:
                meta_comparisons = tuple(reversed(meta_comparisons))

        if ((meta_comparisons[0].split('-')[0] == meta_comparisons[0].split('-')[1])
                and (meta_comparisons[0].split('-')[0] in meta_comparisons[1].split('-'))):
            _, p_value = stats.ttest_ind(
                scores[matrix_shape][meta_comparisons[0]], scores[matrix_shape][meta_comparisons[1]],
                equal_var=False)
            p_values.loc[meta_comparisons[0], meta_comparisons[1]] = p_value

    return p_values


# ------- PLOT SIMILARITY SCORES AND P VALUES FOR CONNECTOMES -------


def colormap(base_color):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_colormap', ['#FFFFFF', base_color])
    return cmap


def plot_legend(ax, hs, ls):
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
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
    'two_clusters_shuffled': '#882255',
    'WS': '#DDCC77',
    'BA': '#EE8866',
}

colors_comparisons = {f'{net_0}-{net_1}': (colors[net_0], colors[net_1])
                      for net_0, net_1 in combinations_with_replacement(networks, 2)}

labels = {
    'ER': 'ER',
    'DCM': 'DCM',
    'one_cluster': '1C',
    'two_clusters': '2C',
    'one_cluster_shuffled': '1Cs',
    'two_clusters_shuffled': '2Cs',
    'WS': 'WS',
    'BA': 'BA',
}


def plot_connectome_similarity(connectomes, matrix_shape, xlims):

    if matrix_shape == 'square':
        mosaic = """
            AAABBB.CCCCX
            AAABBB.CCCCX
            AAABBB.CCCCX
            DDDEEE.FFGGY
            DDDEEE.FFGGY
            DDDEEE.FFGGY
            HHHIII.JJKKZ
            HHHIII.JJKKZ
            HHHIII.JJKKZ
            """
        fig = plt.figure(figsize=(15, 10), layout="constrained", dpi=1200)
        connectome_titles = titles
    elif matrix_shape == 'rectangular':
        mosaic = """
            AABB.CCCCCCX
            AABB.CCCCCCX
            AABB.CCCCCCX
            DDEE.FFFGGGY
            DDEE.FFFGGGY
            DDEE.FFFGGGY
            HHII.JJJKKKZ
            HHII.JJJKKKZ
            HHII.JJJKKKZ
            """
        fig = plt.figure(figsize=(15, 10), layout="constrained", dpi=1200)
        connectome_titles = labels
    ax_dict = fig.subplot_mosaic(mosaic)

    # --- PLOT CONNECTOMES ---
    ax = ax_dict['A']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['ER'], repetitions=1)[0],
                         name='square_ER', title=connectome_titles['ER'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['ER']))
    ax.text(-0.1, 1.1, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['B']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['DCM'], repetitions=1)[0],
                         name='square_DCM', title=connectome_titles['DCM'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['DCM']))
    ax.text(-0.1, 1.1, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['D']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['one_cluster'], repetitions=1)[0],
                         name='square_one_cluster', title=connectome_titles['one_cluster'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['one_cluster']))
    ax.text(-0.1, 1.1, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['E']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['two_clusters'], repetitions=1)[0],
                         name='square_two_clusters', title=connectome_titles['two_clusters'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['two_clusters']))
    ax.text(-0.1, 1.1, 'E', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['H']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['WS'], repetitions=1)[0],
                         name='square_WS', title=connectome_titles['WS'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['WS']))
    ax.text(-0.1, 1.1, 'H', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['I']
    ax = plot_connectome(connectome=singular_angles.draw(connectomes['BA'], repetitions=1)[0],
                         name='square_BA', title=connectome_titles['BA'], fig=fig, ax=ax, save=False,
                         cmap=colormap(colors['BA']))
    ax.text(-0.1, 1.1, 'I', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

    ax = ax_dict['C']
    comparisons = ['ER-ER', 'DCM-DCM', 'one_cluster-one_cluster',
                   'two_clusters-two_clusters', 'BA-BA', 'WS-WS']  # 'ER-DCM']
    ax = singular_angles.plot_similarities(
        similarity_scores={key: scores[matrix_shape][key] for key in comparisons},
        colors=colors_comparisons,
        labels={c: f"{labels[c.split('-')[0]]} - {labels[c.split('-')[1]]}" for c in comparisons},
        ax=ax, legend=False)
    ax.text(-0.1, 1.1, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax.set_xlim(xlims[0])
    hs, ls = ax.get_legend_handles_labels()
    plot_legend(ax_dict['X'], hs, ls)

    ax = ax_dict['F']
    comparisons = ['ER-DCM', 'ER-one_cluster', 'ER-two_clusters', 'ER-BA', 'ER-WS']
    ax = singular_angles.plot_similarities(
        similarity_scores={key: scores[matrix_shape][key] for key in comparisons},
        colors=colors_comparisons,
        labels={c: f"{labels[c.split('-')[0]]} - {labels[c.split('-')[1]]}" for c in comparisons},
        ax=ax, legend=False)
    ax.text(-0.1, 1.1, 'F', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax.set_xlim(xlims[1])
    hs, ls = ax.get_legend_handles_labels()
    ax = ax_dict['G']
    comparisons = ['DCM-one_cluster', 'DCM-two_clusters', 'DCM-BA', 'DCM-WS']
    ax = singular_angles.plot_similarities(
        similarity_scores={key: scores[matrix_shape][key] for key in comparisons},
        colors=colors_comparisons,
        labels={c: f"{labels[c.split('-')[0]]} - {labels[c.split('-')[1]]}" for c in comparisons},
        ax=ax, legend=False)
    ax.text(-0.1, 1.1, 'G', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax.set_xlim(xlims[1])
    hs_, ls_ = ax.get_legend_handles_labels()
    hs += hs_
    ls += ls_
    plot_legend(ax_dict['Y'], hs, ls)

    ax = ax_dict['J']
    comparisons = ['one_cluster-two_clusters', 'one_cluster-BA', 'one_cluster-WS']
    ax = singular_angles.plot_similarities(
        similarity_scores={key: scores[matrix_shape][key] for key in comparisons},
        colors=colors_comparisons,
        labels={c: f"{labels[c.split('-')[0]]} - {labels[c.split('-')[1]]}" for c in comparisons},
        ax=ax, legend=False)
    ax.text(-0.1, 1.1, 'J', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax.set_xlim(xlims[1])
    hs, ls = ax.get_legend_handles_labels()
    ax = ax_dict['K']
    comparisons = ['two_clusters-BA', 'two_clusters-WS', 'WS-BA']
    ax = singular_angles.plot_similarities(
        similarity_scores={key: scores[matrix_shape][key] for key in comparisons},
        colors=colors_comparisons,
        labels={c: f"{labels[c.split('-')[0]]} - {labels[c.split('-')[1]]}" for c in comparisons},
        ax=ax, legend=False)
    ax.text(-0.1, 1.1, 'K', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
    ax.set_xlim(xlims[1])
    hs_, ls_ = ax.get_legend_handles_labels()
    hs += hs_
    ls += ls_
    plot_legend(ax_dict['Z'], hs, ls)

    plt.savefig(f'plots/connectomes_and_similarity_{matrix_shape}.pdf')


def plot_p_values_reduced(p_values, matrix_shape):

    p_values_reduced = np.zeros((len(networks), len(networks)))
    score_labels = np.empty((len(networks), len(networks), 2), dtype=object)

    for i, GT in enumerate(networks):
        comparison_1 = [f'{GT}-{n}' for n in networks]
        # reorder labels of networks
        for k in range(i):
            comparison_1[k] = '-'.join(list(reversed(comparison_1[k].split('-'))))
        p_values_reduced[i, :] = p_values.sel(comparison_0=[f'{n}-{n}' for n in networks][i],
                                              comparison_1=comparison_1)
        score_labels[:, i, 0] = [f'{n}-{n}' for n in networks][i]
        score_labels[:, i, 1] = comparison_1

    np.fill_diagonal(p_values_reduced, np.nan)

    # Define colormap for p-values
    top = cm.get_cmap('Blues', 1000)
    bottom = cm.get_cmap('Reds', 1000)
    newcolors = np.vstack((top(np.linspace(0.95, 0.6, 1000)),
                           bottom(np.linspace(0.45, 0.9, 1000))))
    newcmp = ListedColormap(newcolors)

    n = p_values_reduced.shape[0]
    sig = 0.05 / (n * (n - 1))
    sig_alpha = np.log10(sig)
    newnorm = TwoSlopeNorm(vmin=-50, vcenter=sig_alpha, vmax=0)

    mosaic = """
        AAAAAAAAAAAAAA.B
        """
    fig = plt.figure(figsize=(9, 8), layout="constrained")
    ax_dict = fig.subplot_mosaic(mosaic)

    # Plot color mesh
    ax = ax_dict['A']
    ax.pcolormesh(np.log10(p_values_reduced), cmap=newcmp, norm=newnorm)
    # Add text
    for x in range(n):
        for y in range(n):
            # try:
            #     if np.log10(p_values_reduced[x, y]) > -1:
            #         ax.text(
            #             x + 0.5, y + 0.5, s=f'{np.round(np.log10(p_values_reduced[x, y]), 2)}', va='center',
            #             ha='center', color='white', fontsize=7)
            #     else:
            #         ax.text(x + 0.5, y + 0.5, s=f'{int(np.log10(p_values_reduced[x, y]))}',
            #                 va='center', ha='center', color='white', fontsize=11)
            # except OverflowError:
            #     pass
            # except ValueError:
            #     pass

            ax.text(x + 0.8, y + 0.8, s=f"{np.round(np.mean(scores[matrix_shape][score_labels[x, y, 1]]), 3)}",
                    va='center', ha='center', color='white', fontsize=7)
            ax.text(x + 0.2, y + 0.2, s=f"{np.round(np.mean(scores[matrix_shape][score_labels[x, y, 0]]), 3)}",
                    va='center', ha='center', color='white', fontsize=7)

    # Add white lines around each entry
    ax.set_xticks(np.arange(0, n + 1, step=0.5), minor=True)
    ax.set_yticks(np.arange(0, n + 1, step=0.5), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Format ticks
    ax.set_xticks(np.arange(0.5, n))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0.5, n))
    ax.set_yticklabels([])
    ax.set_xticklabels([f'GT - {labels[n]}' for n in networks],
                       rotation=45, ha='right', va='top')
    ax.set_yticklabels([f'GT = {labels[n]}' for n in networks])
    ax.set_ylabel('self-similarity', size=18)
    ax.set_xlabel('cross-similarity', size=18)
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)

    # Inset for the colorbar
    cb = mpl.colorbar.ColorbarBase(ax_dict['B'], cmap=newcmp,
                                   norm=newnorm,
                                   boundaries=np.arange(-51, 0.1, step=0.1),
                                   # orientation='vertical',
                                   ticks=[-50, -40, -30, -20, -10, sig_alpha, 0])
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(6)
    cb.ax.set_xlabel('log of p-value')
    # cb.ax.set_xlim(-21, -0.1)

    plt.savefig(f'plots/p_values_reduced_{matrix_shape}.pdf')


xlims = {
    'square': ((0.025, 0.10), (0.025, 0.055)),
    'rectangular': ((0.025, 0.16), (0.025, 0.08))
}


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
        sizes=[(300, 300), (200, 450), (100, 900)],
        connectome_types=['ER', 'DCM', 'one_cluster', 'two_clusters', 'WS', 'BA'],
        max_change_fraction=0.1, step_size=0.005, repetitions=5,
        savename='similarity_under_changes',
        log=False):

    mosaic = """
        AAABBBCCCX
        """
    fig = plt.figure(figsize=(int(len(sizes) * 5), 5), layout="constrained")
    ax_dict = fig.subplot_mosaic(mosaic)

    for ax_i, size in zip(['A', 'B', 'C'], sizes):
        ax = ax_dict[ax_i]
        for connectome_type in connectome_types:
            if connectome_type == 'DCM':
                connectome = directed_configuration_model(size, mean_connection_prob)
            elif connectome_type == 'ER':
                connectome = erdos_renyi_connectome(size, mean_connection_prob)
            elif connectome_type == 'one_cluster':
                connectome = clustered_connectome(size=size, clusters=[(0, 50)], rel_cluster_weights=[10],
                                                  mean_connection_prob=mean_connection_prob)
            elif connectome_type == 'two_clusters':
                connectome = clustered_connectome(size, clusters=[(50, 85), (85, 100)],
                                                  rel_cluster_weights=[10, 10],
                                                  mean_connection_prob=mean_connection_prob)
            elif connectome_type == 'WS':
                connectome = watts_strogatz(size, mean_connection_prob)
            elif connectome_type == 'BA':
                connectome = barabasi_albert(size, mean_connection_prob)
            else:
                continue
            similarity_mean, similarity_std, changes, base_matrix = change_matrix(connectome, max_change_fraction,
                                                                                  step_size, repetitions)
            x = changes / (np.shape(base_matrix)[0] * np.shape(base_matrix)[1]) * 100
            ax.plot(x, similarity_mean, color=colors[connectome_type], label=titles[connectome_type])
            ax.fill_between(x, similarity_mean - similarity_std, similarity_mean + similarity_std, alpha=0.3,
                            color=colors[connectome_type])
            ax.set_title(f'matrix size: {size[0]} x {size[1]}')
        ax.set_xlabel('% of changed connections')
        ax.set_ylabel('similarity')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if log:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_yscale('log')
    if log:
        savename += '_log'
    hs, ls = ax.get_legend_handles_labels()
    plot_legend(ax_dict['X'], hs, ls)
    plt.savefig(f'plots/{savename}.png', dpi=600, bbox_inches='tight')


for matrix_shape in ['square', 'rectangular']:
    plot_connectome_similarity(connectome_dict[matrix_shape], matrix_shape, xlims[matrix_shape])
    plot_p_values_reduced(calc_p_values(matrix_shape), matrix_shape)

# calculate_dropoff(max_change_fraction=0.1, step_size=0.001, repetitions=5, log=True)
