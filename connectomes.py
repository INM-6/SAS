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


class Connectomes(SingularAngles):
    """docstring for Connectomes"""

    def __init__(self, size_square, size_rectangular, mean_connection_prob,
                 networks=['ER', 'DCM', 'one_cluster', 'two_clusters', 'WS', 'BA',
                           'one_cluster_shuffled', 'two_clusters_shuffled'], p_WS=0.3):
        super(Connectomes, self).__init__()
        self.p_WS = p_WS
        self.size_square = size_square
        self.size_rectangular = size_rectangular
        self.mean_connection_prob = mean_connection_prob

        self.networks = networks
        self.networks_without_shuffle = [n for n in self.networks if 'shuffle' not in n]

        self.titles = {
            'ER': 'Erdős-Rényi',
            'DCM': 'directed configuration model',
            'one_cluster': 'one cluster',
            'two_clusters': 'two clusters',
            'one_cluster_shuffled': 'one cluster - shuffled',
            'two_clusters_shuffled': 'two clusters - shuffled',
            'WS': 'Watts-Strogatz',
            'BA': 'Barabasi-Albert',
        }

        self.colors = {
            'ER': '#332288',
            'DCM': '#88CCEE',
            'one_cluster': '#44AA99',
            'one_cluster_shuffled': '#999933',
            'two_clusters': '#CC6677',
            'two_clusters_shuffled': '#882255',
            'WS': '#DDCC77',
            'BA': '#EE8866',
        }

        self.labels = {
            'ER': 'ER',
            'DCM': 'DCM',
            'one_cluster': '1C',
            'two_clusters': '2C',
            'one_cluster_shuffled': '1Cs',
            'two_clusters_shuffled': '2Cs',
            'WS': 'WS',
            'BA': 'BA',
        }

        self.mean_num_connections_square = mean_connection_prob * size_square[0] * size_square[1]
        self.mean_num_connections_rectangular = mean_connection_prob * size_rectangular[0] * size_rectangular[1]

        self.create_connectomes()

    # define connectomes

    def create_connectomes(self):
        connectomes_square = {}
        connectomes_rectangular = {}
        if 'ER' in self.networks:
            connectomes_square['ER'] = self.erdos_renyi_connectome(self.size_square)
        connectomes_rectangular['ER'] = self.erdos_renyi_connectome(self.size_rectangular)
        if 'DCM' in self.networks:
            connectomes_square['DCM'] = self.directed_configuration_model(self.size_square)
        connectomes_rectangular['DCM'] = self.directed_configuration_model(self.size_rectangular)
        if 'one_cluster' in self.networks:
            connectomes_square['one_cluster'] = self.clustered_connectome(
                self.size_square, clusters=[(0, 50)], rel_cluster_weights=[10])
            connectomes_rectangular['one_cluster'] = self.clustered_connectome(
                self.size_rectangular, clusters=[(0, 50)], rel_cluster_weights=[10])
            connectomes_square['one_cluster_shuffled'] = self._shuffle(connectomes_square['one_cluster'])
            connectomes_rectangular['one_cluster_shuffled'] = self._shuffle(connectomes_rectangular['one_cluster'])
        if 'two_clusters' in self.networks:
            connectomes_square['two_clusters'] = self.clustered_connectome(
                self.size_square, clusters=[(50, 85), (85, 100)], rel_cluster_weights=[10, 10])
            connectomes_rectangular['two_clusters'] = self.clustered_connectome(
                self.size_rectangular, clusters=[(50, 85), (85, 100)], rel_cluster_weights=[10, 10])
            connectomes_square['two_clusters_shuffled'] = self._shuffle(connectomes_square['two_clusters'])
            connectomes_rectangular['two_clusters_shuffled'] = self._shuffle(connectomes_rectangular['two_clusters'])
        if 'ER' in self.networks:
            connectomes_square['WS'] = self.watts_strogatz(self.size_square)
            connectomes_rectangular['WS'] = self.watts_strogatz(self.size_rectangular, p=0.3)
        if 'ER' in self.networks:
            connectomes_square['BA'] = self.barabasi_albert(self.size_square)
            connectomes_rectangular['BA'] = self.barabasi_albert(self.size_rectangular)

        self.connectome_dict = {'square': connectomes_square, 'rectangular': connectomes_rectangular}

    def clustered_connectome(self, size, clusters, rel_cluster_weights):
        mean_num_connections = mean_connection_prob * size[0] * size[1]
        connectome = np.ones(size)
        for cluster, rel_cluster_weight in zip(clusters, rel_cluster_weights):
            connectome[cluster[0]:cluster[1], :][:, cluster[0]:cluster[1]] = rel_cluster_weight
        return connectome / (np.sum(connectome) / mean_num_connections)

    def erdos_renyi_connectome(self, size):
        return np.ones(size) * mean_connection_prob

    def directed_configuration_model(self, size, np_seed=1):

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

    def watts_strogatz(self, size, p=0.3):
        graph = nx.watts_strogatz_graph(np.maximum(size[0], size[1]), k=int(
            mean_connection_prob * (np.maximum(size[0], size[1]) - 1)), p=p)
        matrix = nx.to_numpy_array(graph).T
        if size[0] != size[1]:
            matrix = self._rectangularize(matrix, size)
        return matrix

    def barabasi_albert(self, size):
        graph = nx.barabasi_albert_graph(np.maximum(size[0], size[1]), m=int(
            mean_connection_prob * (np.maximum(size[0], size[1]) - 1) / 2))
        matrix = nx.to_numpy_array(graph).T
        if size[0] != size[1]:
            matrix = self._rectangularize(matrix, size)
        return matrix

    def _shuffle(self, matrix, np_seed=1):
        np.random.seed(np_seed)
        n, m = matrix.shape
        flat_matrix = matrix.flatten()
        np.random.shuffle(flat_matrix)
        return flat_matrix.reshape(n, m)

    def _rectangularize(self, matrix, size):
        random_choice = np.random.choice(np.maximum(size[0], size[1]), size=np.minimum(size[0], size[1]), replace=False)
        if size[0] < size[1]:
            matrix = matrix[random_choice, :]
        else:
            matrix = matrix[:, random_choice]
        return matrix

    # plotting

    def plot_connectome(self, connectome, name, title, fig=None, ax=None, save=True, cmap='Greens'):
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

    def calc_p_values(self, scores):
        # calculate p values between distributions
        comparisons = []
        for i, network_i in enumerate(self.networks):
            for j, network_j in enumerate(self.networks):
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
                    if (self.networks.index(meta_comparisons[1].split('-')[0])
                            < self.networks.index(meta_comparisons[0].split('-')[0])):
                        meta_comparisons = tuple(reversed(meta_comparisons))
                else:
                    meta_comparisons = tuple(reversed(meta_comparisons))

            if ((meta_comparisons[0].split('-')[0] == meta_comparisons[0].split('-')[1])
                    and (meta_comparisons[0].split('-')[0] in meta_comparisons[1].split('-'))):
                _, p_value = stats.ttest_ind(
                    scores[meta_comparisons[0]], scores[meta_comparisons[1]],
                    equal_var=False)
                p_values.loc[meta_comparisons[0], meta_comparisons[1]] = p_value

        return p_values

    # ------- PLOT SIMILARITY SCORES AND P VALUES FOR CONNECTOMES -------

    def colormap(self, base_color):
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'custom_colormap', ['#FFFFFF', base_color])
        return cmap

    def plot_legend(self, ax, hs, ls, loc=(0.1, 0.1), fontsize=9):
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(hs, ls, frameon=False, loc=loc, ncol=1, fontsize=fontsize)

    def plot_connectome_similarity(self, connectomes, matrix_shape):

        if matrix_shape == 'square':
            mosaic = """
                AAAaaa.BBBbbb
                AAAaaa.BBBbbb
                AAAaaa.BBBbbb
                CCCccc.DDDddd
                CCCccc.DDDddd
                CCCccc.DDDddd
                EEEeee.FFFfff
                EEEeee.FFFfff
                EEEeee.FFFfff
                """
            fig = plt.figure(figsize=(15, 10), layout="constrained", dpi=1200)
            connectome_titles = self.titles
        elif matrix_shape == 'rectangular':
            mosaic = """
                AAAaaa.BBBbbb
                AAAaaa.BBBbbb
                AAAaaa.BBBbbb
                CCCccc.DDDddd
                CCCccc.DDDddd
                CCCccc.DDDddd
                EEEeee.FFFfff
                EEEeee.FFFfff
                EEEeee.FFFfff
                """
            fig = plt.figure(figsize=(11, 10), layout="constrained", dpi=1200)
            connectome_titles = self.labels
        ax_dict = fig.subplot_mosaic(mosaic)

        # --- PLOT CONNECTOMES ---
        ax = ax_dict['A']
        ax = self.plot_connectome(connectome=self.draw(connectomes['ER'], repetitions=1)[0],
                                  name='square_ER', title=connectome_titles['ER'], fig=fig, ax=ax, save=False,
                                  cmap=self.colormap(self.colors['ER']))
        ax.text(-0.1, 1.15, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['B']
        ax = self.plot_connectome(connectome=self.draw(connectomes['DCM'], repetitions=1)[0],
                                  name='square_DCM', title=connectome_titles['DCM'], fig=fig, ax=ax, save=False,
                                  cmap=self.colormap(self.colors['DCM']))
        ax.text(-0.1, 1.15, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['C']
        ax = self.plot_connectome(connectome=self.draw(connectomes['one_cluster'], repetitions=1)[0],
                                  name='square_one_cluster', title=connectome_titles['one_cluster'], fig=fig, ax=ax,
                                  save=False,
                                  cmap=self.colormap(self.colors['one_cluster']))
        ax.text(-0.1, 1.15, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['D']
        ax = self.plot_connectome(connectome=self.draw(connectomes['two_clusters'], repetitions=1)[0],
                                  name='square_two_clusters', title=connectome_titles['two_clusters'], fig=fig, ax=ax,
                                  save=False,
                                  cmap=self.colormap(self.colors['two_clusters']))
        ax.text(-0.1, 1.15, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['E']
        ax = self.plot_connectome(connectome=self.draw(connectomes['WS'], repetitions=1)[0],
                                  name='square_WS', title=connectome_titles['WS'], fig=fig, ax=ax, save=False,
                                  cmap=self.colormap(self.colors['WS']))
        ax.text(-0.1, 1.15, 'E', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['F']
        ax = self.plot_connectome(connectome=self.draw(connectomes['BA'], repetitions=1)[0],
                                  name='square_BA', title=connectome_titles['BA'], fig=fig, ax=ax, save=False,
                                  cmap=self.colormap(self.colors['BA']))
        ax.text(-0.1, 1.15, 'F', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        for GT, axid in zip(self.networks_without_shuffle, ['a', 'b', 'c', 'd', 'e', 'f']):
            ax = ax_dict[axid]
            ax.text(-0.1, 1.15, axid, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

            ax.hist(scores[matrix_shape][f'{GT}-{GT}'], density=True, edgecolor=self.colors[GT], color=self.colors[GT],
                    histtype="stepfilled", linewidth=1.5)
            for network in self.networks_without_shuffle:
                if network != GT:
                    try:
                        score = scores[matrix_shape][f'{GT}-{network}']
                    except KeyError:
                        score = scores[matrix_shape][f'{network}-{GT}']
                    ax.hist(score, density=True, edgecolor=self.colors[network], histtype="step", linewidth=3)
            ax.set_ylabel('occurrence')
            ax.set_xlabel('similarity')
        plt.savefig(f'plots/connectomes_and_similarity_{matrix_shape}.pdf')

    def plot_p_values_reduced(self, p_values, scores, matrix_shape, savename):

        p_values_reduced = np.zeros((len(self.networks), len(self.networks)))
        score_labels = np.empty((len(self.networks), len(self.networks), 2), dtype=object)

        for i, GT in enumerate(self.networks):
            comparison_1 = [f'{GT}-{n}' for n in self.networks]
            # reorder labels of networks
            for k in range(i):
                comparison_1[k] = '-'.join(list(reversed(comparison_1[k].split('-'))))
            p_values_reduced[i, :] = p_values.sel(comparison_0=[f'{n}-{n}' for n in self.networks][i],
                                                  comparison_1=comparison_1)
            score_labels[:, i, 0] = [f'{n}-{n}' for n in self.networks][i]
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
            AAAAAAAAAAAAAAAAAA.B
            """
        fig = plt.figure(figsize=(9, 8), layout="constrained")
        ax_dict = fig.subplot_mosaic(mosaic)

        # Plot color mesh
        ax = ax_dict['A']
        ax.pcolormesh(np.log10(p_values_reduced), cmap=newcmp, norm=newnorm, edgecolor='white')
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

                ax.text(x + 0.65, y + 0.65, s=f"{np.round(np.mean(scores[matrix_shape][score_labels[x, y, 1]]), 3)}",
                        va='center', ha='center', color='white', fontsize=7)
                ax.text(x + 0.35, y + 0.35, s=f"{np.round(np.mean(scores[matrix_shape][score_labels[x, y, 0]]), 3)}",
                        va='center', ha='center', color='white', fontsize=7)

        # Add white lines around each entry
        ax.set_xticks(np.arange(0, n + 1, step=0.5), minor=True)
        ax.set_yticks(np.arange(0, n + 1, step=0.5), minor=True)
        # ax.grid(visible=True, which='minor', color='white', linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        # Format ticks
        ax.set_xticks(np.arange(0.5, n))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0.5, n))
        ax.set_yticklabels([])
        ax.set_xticklabels([f'GT - {self.labels[n]}' for n in self.networks],
                           rotation=45, ha='right', va='top')
        ax.set_yticklabels([f'GT = {self.labels[n]}' for n in self.networks])
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

        plt.savefig(f'plots/{savename}.pdf')

    # --- CALCULATE SIMILARITY FOR INCREASINGLY DIFFERENT MATRICES

    def change_matrix(self, base_matrix_template, max_change_fraction=0.1, step_size=0.01, repetitions=10):

        base_matrix = self.draw(base_matrix_template, repetitions=1)[0]
        size = int(np.shape(base_matrix)[0] * np.shape(base_matrix)[1])
        max_changes = int(size * max_change_fraction)

        changes = []
        similarity_mean = []
        similarity_std = []
        for change in np.arange(max_changes + 1, step=size * step_size).astype(int):
            for change in np.insert(np.arange(max_changes + 1, step=size * step_size).astype(int), 1, np.arange(1, 100, step=10)):
                similarity = []
                for rep in range(repetitions):
                    changed_matrix = base_matrix.copy()
                    indices = list(product(np.arange(np.shape(changed_matrix)[
                                   0]), np.arange(np.shape(changed_matrix)[1])))
                    if change > 0:
                        samples = random.sample(indices, change)
                        for coordinate in samples:
                            if changed_matrix[coordinate] == 0:
                                changed_matrix[coordinate] += 1
                            else:
                                changed_matrix[coordinate] -= 1
                    similarity.append(self.compare(base_matrix, changed_matrix))
                similarity_mean.append(np.mean(similarity))
                similarity_std.append(np.std(similarity))
                changes.append(change)

        similarity_mean = np.array(similarity_mean)
        similarity_std = np.array(similarity_std)
        changes = np.array(changes)

        return similarity_mean, similarity_std, changes, base_matrix

    def calculate_dropoff(
            self,
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
                    connectome = self.directed_configuration_model(size)
                elif connectome_type == 'ER':
                    connectome = self.erdos_renyi_connectome(size)
                elif connectome_type == 'one_cluster':
                    connectome = self.clustered_connectome(size=size, clusters=[(0, 50)], rel_cluster_weights=[10])
                elif connectome_type == 'two_clusters':
                    connectome = self.clustered_connectome(size, clusters=[(50, 85), (85, 100)],
                                                           rel_cluster_weights=[10, 10])
                elif connectome_type == 'WS':
                    connectome = self.watts_strogatz(size)
                elif connectome_type == 'BA':
                    connectome = self.barabasi_albert(size)
                else:
                    continue
                similarity_mean, similarity_std, changes, base_matrix = self.change_matrix(connectome, max_change_fraction,
                                                                                           step_size, repetitions)
                x = changes / (np.shape(base_matrix)[0] * np.shape(base_matrix)[1]) * 100
                ax.plot(x, similarity_mean, color=self.colors[connectome_type], label=self.titles[connectome_type])
                ax.fill_between(x, similarity_mean - similarity_std, similarity_mean + similarity_std, alpha=0.3,
                                color=self.colors[connectome_type])
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
        self.plot_legend(ax_dict['X'], hs, ls)
        plt.savefig(f'plots/{savename}.png', dpi=600, bbox_inches='tight')

    def plot_p_increase(self, scores, savename):
        fig, ax = plt.subplots()
        for network in self.networks_without_shuffle:
            scores_arr = scores[f'{GT_increase}-{network}']
            mean, std = np.empty(len(scores_arr)), np.empty(len(scores_arr))
            for i, obj in enumerate(scores_arr):
                mean[i] = np.mean(obj)
                std[i] = np.std(obj)
            ax.plot(increase, mean, color=self.colors[network])
            ax.fill_between(increase, mean - std, mean + std, color=self.colors[network], alpha=0.5)
        ax.set_xlabel('number of draws')
        ax.set_ylabel('similarity')
        plt.savefig(f'plots/{savename}.pdf')


if __name__ == '__main__':

    size_square = (300, 300)
    size_rectangular = (450, 200)
    mean_connection_prob = 0.1
    repetitions = 100

    connectomes = Connectomes(size_square=size_square, size_rectangular=size_rectangular,
                              mean_connection_prob=mean_connection_prob)

    # plot instantiated connectomes
    os.makedirs('plots', exist_ok=True)
    # compare all connectomes with each other

    score_name = 'scores'

    # 100 vs 100
    try:
        scores = np.load(f'{score_name}.npy', allow_pickle=True).item()
        print('Scores found on disk. Continuing...')
    except FileNotFoundError:
        print('Scores not found on disk. Calculating...')
        scores = {}
        for matrix_shape, connectomes in connectomes.connectome_dict.items():
            scores[matrix_shape] = {}
            for rule_1, rule_2 in combinations_with_replacement(connectomes.keys(), 2):
                score = connectomes.similarity(connectomes[rule_1], connectomes[rule_2],
                                               repetitions=repetitions)
                scores[matrix_shape][f'{rule_1}-{rule_2}'] = score
                print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(score), 2)} ± "
                      f"{np.round(np.std(score), 2)}")
        np.save(f'{score_name}.npy', scores)

    # 1 vs 100
    try:
        scores_GT = np.load(f'{score_name}_GT.npy', allow_pickle=True).item()
        print('Scores (GT) found on disk. Continuing...')
    except FileNotFoundError:
        print('Scores (GT) not found on disk. Calculating...')
        scores_GT = {}
        for matrix_shape, connectomes in connectomes.connectome_dict.items():
            scores_GT[matrix_shape] = {}
            for rule_1, rule_2 in product(connectomes.keys(), connectomes.keys()):
                score = connectomes.singular_angles.similarity(connectomes[rule_1], connectomes[rule_2],
                                                               repetitions=repetitions, repeat_a=True)
                scores_GT[matrix_shape][f'{rule_1}-{rule_2}'] = score
                print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(score), 2)} ± "
                      f"{np.round(np.std(score), 2)}")
        np.save(f'{score_name}_GT.npy', scores_GT)

    # 1 vs x
    GT_increase = 'ER'
    increase = np.arange(3, 100, 5)
    try:
        scores_GT_increase = np.load(f'{score_name}_GT_increase.npy', allow_pickle=True).item()
        print('Scores (GT increase) found on disk. Continuing...')
    except FileNotFoundError:
        print('Scores (GT increase) not found on disk. Calculating...')
        scores_GT_increase = {}
        for matrix_shape, connectomes in connectomes.connectome_dict.items():
            scores_GT_increase[matrix_shape] = {}
            for rule_2 in connectomes.keys():
                scores_GT_increase[matrix_shape][f'{GT_increase}-{rule_2}'] = np.empty(len(increase), dtype=object)
                for i, incrs in enumerate(increase):
                    scores_GT_increase[matrix_shape][f'{GT_increase}-{rule_2}'][i] = connectomes.similarity(
                        connectomes[GT_increase], connectomes[rule_2], repetitions=incrs, repeat_a=True)
                print(f'calculated {GT_increase}-{rule_2}')
        np.save(f'{score_name}_GT_increase.npy', scores_GT_increase)

    for matrix_shape in ['square', 'rectangular']:
        connectomes.plot_connectome_similarity(connectomes.connectome_dict[matrix_shape], matrix_shape)
        connectomes.plot_p_values_reduced(connectomes.calc_p_values(scores=scores[matrix_shape]), scores=scores,
                                          matrix_shape=matrix_shape, savename=f'p_values_reduced_{matrix_shape}')
        connectomes.plot_p_values_reduced(connectomes.calc_p_values(scores=scores_GT[matrix_shape]), scores=scores_GT,
                                          matrix_shape=matrix_shape, savename=f'p_values_reduced_{matrix_shape}_GT')
        connectomes.plot_p_increase(scores_GT_increase[matrix_shape], savename=f'p_value_increase_{matrix_shape}')

    connectomes.calculate_dropoff(max_change_fraction=0.1, step_size=0.005, repetitions=10, log=True)
