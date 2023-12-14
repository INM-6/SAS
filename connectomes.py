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

    def __init__(self, network_params):
        super(Connectomes, self).__init__()

        self.params = network_params

        self.networks = ['ER', 'DCM', '1C', '2C', 'WS', 'BA']
        self.networks_without_shuffle = [n for n in self.networks if 's' not in n]
        self.matrix_shapes = ['square', 'rectangular']

        self.titles = {
            'ER': 'Erdős-Rényi',
            'DCM': 'directed configuration model',
            '1C': 'one cluster',
            '2C': 'two clusters',
            '1Cs': 'one cluster - shuffled',
            '2Cs': 'two clusters - shuffled',
            'WS': 'Watts-Strogatz',
            'BA': 'Barabasi-Albert',
        }

        self.colors = {
            'ER': '#332288',
            'DCM': '#88CCEE',
            '1C': '#44AA99',
            '1Cs': '#999933',
            '2C': '#CC6677',
            '2Cs': '#882255',
            'WS': '#DDCC77',
            'BA': '#EE8866',
        }

        self.labels = {
            'ER': 'ER',
            'DCM': 'DCM',
            '1C': '1C',
            '2C': '2C',
            '1Cs': '1Cs',
            '2Cs': '2Cs',
            'WS': 'WS',
            'BA': 'BA',
        }

        # # not used !?
        # self.mean_num_connections = {ms: (self.params['mean_connection_prob']
        #                                   * self.params['size'][ms][0]
        #                                   * self.params['size'][ms][1]) for ms in self.matrix_shapes}

        if 'DCM' in self.networks:
            self.indegrees = {}
            self.outdegrees = {}

        self.deleted_source_ids = {}

    def clustered_connectome(self, size, clusters, rel_cluster_weights):
        max_size = np.max(size)
        mean_num_connections = self.params['mean_connection_prob'] * max_size * max_size
        connectome = np.ones((max_size, max_size))
        for cluster, rel_cluster_weight in zip(clusters, rel_cluster_weights):
            print(cluster)
            connectome[cluster[0]:cluster[1], :][:, cluster[0]:cluster[1]] = rel_cluster_weight
        connectome = connectome / (np.sum(connectome) / mean_num_connections)
        matrix = (np.random.random((max_size, max_size)) < connectome).astype(int)

        if size[0] != size[1]:
            matrix = self._rectangularize(matrix, size)
        return matrix

    def erdos_renyi(self, size):
        max_size = np.max(size)
        matrix = (np.random.random((max_size, max_size))
                  < (np.ones(max_size, max_size) * self.params['mean_connection_prob'])).astype(int)

        if size[0] != size[1]:
            matrix = self._rectangularize(matrix, size)
        return matrix

    def directed_configuration_model(self, size):
        max_size = np.max(size)
        total_connections = int(self.params['mean_connection_prob'] * max_size * max_size)

        def calc_degrees(length, total_connections):
            connections = np.ones(total_connections)
            degrees = np.zeros(length)
            np.add.at(degrees, np.random.choice(range(length), total_connections, replace=True), connections)
            return degrees.astype(int)

        if (max_size not in self.indegrees.keys()) and (max_size not in self.outdegrees.keys()):
            self.indegrees[max_size] = calc_degrees(max_size, total_connections)
            self.outdegrees[max_size] = calc_degrees(max_size, total_connections)

        graph = nx.directed_configuration_model(
            in_degree_sequence=self.indegrees[max_size], out_degree_sequence=self.outdegrees[max_size])
        graph.remove_edges_from(nx.selfloop_edges(graph))
        while graph.number_of_edges() < total_connections:
            nodes = np.random.choice(graph.nodes(), size=2, replace=False)
            graph.add_edge(nodes[0], nodes[1])
        while graph.number_of_edges() > total_connections:
            edge = np.random.choice(graph.edges())
            graph.remove_edge(*edge)
        matrix = nx.to_numpy_array(graph)

        if size[0] != size[1]:
            matrix = self._rectangularize(matrix, size)
        return matrix

    def watts_strogatz(self, size, p):
        max_size = np.max(size)
        graph = nx.watts_strogatz_graph(max_size, k=int(self.params['mean_connection_prob'] * (max_size - 1)),
                                        p=p)
        matrix = nx.to_numpy_array(graph).T
        if size[0] != size[1]:
            matrix = self._rectangularize(matrix, size)
        return matrix

    def barabasi_albert(self, size):
        max_size = np.max(size)
        graph = nx.barabasi_albert_graph(max_size, m=int(self.params['mean_connection_prob'] * (max_size - 1) / 2))
        matrix = nx.to_numpy_array(graph).T
        if size[0] != size[1]:
            matrix = self._rectangularize(matrix, size)
        return matrix

    def _shuffle(self, matrix):
        n, m = matrix.shape
        flat_matrix = matrix.flatten()
        np.random.shuffle(flat_matrix)
        return flat_matrix.reshape(n, m)

    def _rectangularize(self, matrix, size):
        try:
            random_choice = self.deleted_source_ids[size]
        except KeyError:
            random_choice = np.sort(np.random.choice(np.maximum(size[0], size[1]),
                                                     size=np.minimum(size[0], size[1]), replace=False))
            self.deleted_source_ids[size] = random_choice
        matrix = matrix[:, random_choice]
        return matrix

    # plotting

    def plot_connectome(self, connectome, name, title, fig=None, ax=None, save=True, cmap='Greens'):
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        im = ax.imshow(connectome, cmap=cmap, vmin=0, vmax=np.max(connectome))
        cbar = fig.colorbar(im, ax=ax)
        ax.set_xlabel('source node')
        ax.set_ylabel('target node')
        ax.set_title(title)
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if save:
            plt.savefig(f'plots/{name}.png', dpi=600)
        return ax

    def calc_statistics(self, scores):
        # calculate p values and effect sizes between distributions
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
        effect_sizes = xr.DataArray(
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
                effect_size = ((np.mean(scores[meta_comparisons[0]]) - np.mean(scores[meta_comparisons[1]]))
                               / (np.sqrt((np.std(scores[meta_comparisons[0]])**2
                                           + np.std(scores[meta_comparisons[1]])**2) / 2)))
                p_values.loc[meta_comparisons[0], meta_comparisons[1]] = p_value
                effect_sizes.loc[meta_comparisons[0], meta_comparisons[1]] = effect_size

        return p_values, effect_sizes

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

    def plot_connectome_similarity(self, scores, matrix_shape):

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
        ax = self.plot_connectome(connectome=self.erdos_renyi(self.params['size'][matrix_shape]),
                                  name='square_ER', title=connectome_titles['ER'], fig=fig, ax=ax, save=False,
                                  cmap=self.colormap(self.colors['ER']))
        ax.text(-0.1, 1.15, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['B']
        ax = self.plot_connectome(connectome=self.directed_configuration_model(self.params['size'][matrix_shape]),
                                  name='square_DCM', title=connectome_titles['DCM'], fig=fig, ax=ax, save=False,
                                  cmap=self.colormap(self.colors['DCM']))
        ax.text(-0.1, 1.15, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['C']
        ax = self.plot_connectome(connectome=self.clustered_connectome(self.params['size'][matrix_shape],
                                                                       self.params['1C']['clusters'],
                                                                       self.params['1C']['rel_cluster_weights']),
                                  name='square_1C', title=connectome_titles['1C'], fig=fig, ax=ax,
                                  save=False,
                                  cmap=self.colormap(self.colors['1C']))
        ax.text(-0.1, 1.15, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['D']
        ax = self.plot_connectome(connectome=self.clustered_connectome(self.params['size'][matrix_shape],
                                                                       self.params['2C']['clusters'],
                                                                       self.params['2C']['rel_cluster_weights']),
                                  name='square_2C', title=connectome_titles['2C'], fig=fig, ax=ax,
                                  save=False,
                                  cmap=self.colormap(self.colors['2C']))
        ax.text(-0.1, 1.15, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['E']
        ax = self.plot_connectome(connectome=self.watts_strogatz(self.params['size'][matrix_shape],
                                                                 p=self.params['WS']['p']),
                                  name='square_WS', title=connectome_titles['WS'], fig=fig, ax=ax, save=False,
                                  cmap=self.colormap(self.colors['WS']))
        ax.text(-0.1, 1.15, 'E', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        ax = ax_dict['F']
        ax = self.plot_connectome(connectome=self.barabasi_albert(self.params['size'][matrix_shape]),
                                  name='square_BA', title=connectome_titles['BA'], fig=fig, ax=ax, save=False,
                                  cmap=self.colormap(self.colors['BA']))
        ax.text(-0.1, 1.15, 'F', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

        for GT, axid in zip(self.networks_without_shuffle, ['a', 'b', 'c', 'd', 'e', 'f']):
            ax = ax_dict[axid]
            ax.text(-0.1, 1.15, axid, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

            ax.hist(scores[f'{GT}-{GT}'], density=True, edgecolor=self.colors[GT], color=self.colors[GT],
                    histtype="stepfilled", linewidth=1.5)
            for network in self.networks_without_shuffle:
                if network != GT:
                    try:
                        score = scores[f'{GT}-{network}']
                    except KeyError:
                        score = scores[f'{network}-{GT}']
                    ax.hist(score, density=True, edgecolor=self.colors[network], histtype="step", linewidth=3)
            ax.set_ylabel('occurrence')
            ax.set_xlabel('SAS')
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
                ax.text(x + 0.65, y + 0.65, s=f"{np.round(np.mean(scores[score_labels[x, y, 1]]), 3)}",
                        va='center', ha='center', color='white', fontsize=7)
                ax.text(x + 0.35, y + 0.35, s=f"{np.round(np.mean(scores[score_labels[x, y, 0]]), 3)}",
                        va='center', ha='center', color='white', fontsize=7)

        # Add white lines around each entry
        ax.set_xticks(np.arange(0, n + 1, step=0.5), minor=True)
        ax.set_yticks(np.arange(0, n + 1, step=0.5), minor=True)
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

    def plot_effect_sizes(self, effect_sizes, scores, matrix_shape, savename):

        effect_sizes_reduced = np.zeros((len(self.networks), len(self.networks)))
        score_labels = np.empty((len(self.networks), len(self.networks), 2), dtype=object)

        for i, GT in enumerate(self.networks):
            comparison_1 = [f'{GT}-{n}' for n in self.networks]
            # reorder labels of networks
            for k in range(i):
                comparison_1[k] = '-'.join(list(reversed(comparison_1[k].split('-'))))
            effect_sizes_reduced[i, :] = effect_sizes.sel(comparison_0=[f'{n}-{n}' for n in self.networks][i],
                                                          comparison_1=comparison_1)
            score_labels[:, i, 0] = [f'{n}-{n}' for n in self.networks][i]
            score_labels[:, i, 1] = comparison_1

        np.fill_diagonal(effect_sizes_reduced, np.nan)

        # Define colormap for p-values
        top = cm.get_cmap('Blues', 1000)
        bottom = cm.get_cmap('Reds', 1000)
        newcolors = np.vstack((top(np.linspace(0.95, 0.4, 1000)),
                               bottom(np.linspace(0.4, 0.95, 1000))))
        newcmp = ListedColormap(newcolors)

        n = effect_sizes_reduced.shape[0]
        newnorm = TwoSlopeNorm(vmin=-1, vcenter=1, vmax=10)

        mosaic = """
            AAAAAAAAAAAAAAAAAA.B
            """
        fig = plt.figure(figsize=(9, 8), layout="constrained")
        ax_dict = fig.subplot_mosaic(mosaic)

        # Plot color mesh
        ax = ax_dict['A']
        ax.pcolormesh(effect_sizes_reduced, cmap=newcmp, norm=newnorm, edgecolor='white')

        # Add white lines around each entry
        ax.set_xticks(np.arange(0, n + 1, step=0.5), minor=True)
        ax.set_yticks(np.arange(0, n + 1, step=0.5), minor=True)
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
                                       boundaries=np.arange(-1, 10, step=0.1),
                                       # orientation='vertical',
                                       ticks=[-1, 0, 1, 5, 10])
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(6)
        cb.ax.set_xlabel('effect size')
        # cb.ax.set_xlim(-21, -0.1)

        plt.savefig(f'plots/{savename}.pdf')

    # --- CALCULATE SIMILARITY FOR INCREASINGLY DIFFERENT MATRICES

    def change_matrix(self, base_matrix, max_change_fraction=0.1, step_size=0.01, repetitions=5):

        size = int(np.shape(base_matrix)[0] * np.shape(base_matrix)[1])
        max_changes = int(size * max_change_fraction)

        changes = []
        similarity_mean = []
        similarity_std = []
        for change in np.insert(np.arange(max_changes + 1, step=size * step_size).astype(int),
                                1, np.arange(1, 100, step=10)):
            similarity = []
            for rep in range(repetitions):
                changed_matrix = base_matrix.copy()
                indices = list(product(np.arange(np.shape(changed_matrix)[0]),
                                       np.arange(np.shape(changed_matrix)[1])))
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
            connectome_types=['ER', 'DCM', '1C', '2C', 'WS', 'BA'],
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
                    connectome = self.erdos_renyi(size)
                elif connectome_type == '1C':
                    connectome = self.clustered_connectome(size=size, clusters=self.params['1C']['clusters'],
                                                           rel_cluster_weights=self.params['1C']['rel_cluster_weights'])
                elif connectome_type == '2C':
                    connectome = self.clustered_connectome(size=size, clusters=self.params['2C']['clusters'],
                                                           rel_cluster_weights=self.params['2C']['rel_cluster_weights'])
                elif connectome_type == 'WS':
                    connectome = self.watts_strogatz(size, self.params['WS']['p'])
                elif connectome_type == 'BA':
                    connectome = self.barabasi_albert(size)
                else:
                    continue
                similarity_mean, similarity_std, changes, base_matrix = self.change_matrix(
                    connectome, max_change_fraction, step_size, repetitions)
                x = changes / (np.shape(base_matrix)[0] * np.shape(base_matrix)[1]) * 100
                ax.plot(x, similarity_mean, color=self.colors[connectome_type], label=self.titles[connectome_type])
                ax.fill_between(x, similarity_mean - similarity_std, similarity_mean + similarity_std, alpha=0.3,
                                color=self.colors[connectome_type])
                ax.set_title(f'matrix shape: {size[0]} x {size[1]}')
            ax.set_xlabel('% of changed connections')
            ax.set_ylabel('SAS')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if log:
                # ax.set_xscale('log')
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.set_yscale('log')
        if log:
            savename += '_log'
        hs, ls = ax.get_legend_handles_labels()
        self.plot_legend(ax_dict['X'], hs, ls)
        plt.savefig(f'plots/{savename}.png', dpi=600, bbox_inches='tight')

    def plot_p_increase(self, scores, savename, GT_increase='ER', increase=np.arange(3, 100, 5)):
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
        ax.set_ylabel('SAS')
        plt.savefig(f'plots/{savename}.pdf')

    def draw(self, name, matrix_shape):

        if name == 'ER':
            return self.erdos_renyi(self.params['size'][matrix_shape])
        if name == 'DCM':
            return self.directed_configuration_model(self.params['size'][matrix_shape])
        if name == '1C':
            return self.clustered_connectome(self.params['size'][matrix_shape], self.params['1C']['clusters'],
                                             self.params['1C']['rel_cluster_weights'])
        if name == '2C':
            return self.clustered_connectome(self.params['size'][matrix_shape], self.params['2C']['clusters'],
                                             self.params['2C']['rel_cluster_weights'])
        if name == 'WS':
            return self.watts_strogatz(self.params['size'][matrix_shape], self.params['WS']['p'])
        if name == 'BA':
            return self.barabasi_albert(self.params['size'][matrix_shape])
        if name == '1Cs':
            graph = self.clustered_connectome(self.params['size'][matrix_shape], self.params['1C']['clusters'],
                                              self.params['1C']['rel_cluster_weights'])
            return self._shuffle(graph)
        if name == '2Cs':
            graph = self.clustered_connectome(self.params['size'][matrix_shape], self.params['2C']['clusters'],
                                              self.params['2C']['rel_cluster_weights'])
            return self._shuffle(graph)

    def compare_networks(self, matrix_shape, score_name='scores', GT_increase='ER', increase=np.arange(3, 100, 5)):

        # 100 vs 100
        try:
            scores = np.load(f'{score_name}_{matrix_shape}.npy', allow_pickle=True).item()
            print('Scores found on disk. Continuing...')
        except FileNotFoundError:
            print('Scores not found on disk. Calculating...')
            scores = {}
            for rule_1, rule_2 in combinations_with_replacement(self.networks, 2):
                score = [self.compare(self.draw(rule_1, matrix_shape),
                                      self.draw(rule_2, matrix_shape)) for _ in range(repetitions)]
                scores[f'{rule_1}-{rule_2}'] = score
                print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(score), 2)} ± "
                      f"{np.round(np.std(score), 2)}")
            np.save(f'{score_name}_{matrix_shape}.npy', scores)

        # 1 vs 100
        # try:
        #     scores_GT = np.load(f'{score_name}_GT_{matrix_shape}.npy', allow_pickle=True).item()
        #     print('Scores (GT) found on disk. Continuing...')
        # except FileNotFoundError:
        #     print('Scores (GT) not found on disk. Calculating...')
        #     scores_GT = {}
        #     for rule_1, rule_2 in product(self.networks, self.networks):
        #         connectome_1 = self.draw(rule_1, matrix_shape)
        #         score = [self.compare(connectome_1, self.draw(rule_2, matrix_shape)) for _ in range(repetitions)]
        #         scores_GT[f'{rule_1}-{rule_2}'] = score
        #         print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(score), 2)} ± "
        #               f"{np.round(np.std(score), 2)}")
        #     np.save(f'{score_name}_GT_{matrix_shape}.npy', scores_GT)

        # 1 vs x
        # try:
        #     scores_GT_increase = np.load(f'{score_name}_GT_increase_{matrix_shape}.npy', allow_pickle=True).item()
        #     print('Scores (GT increase) found on disk. Continuing...')
        # except FileNotFoundError:
        #     print('Scores (GT increase) not found on disk. Calculating...')
        #     connectome_GT = self.draw(GT_increase, matrix_shape)
        #     scores_GT_increase = {}
        #     for rule_2 in self.networks:
        #         scores_GT_increase[f'{GT_increase}-{rule_2}'] = np.empty(len(increase), dtype=object)
        #         for i, incrs in enumerate(increase):
        #             scores_GT_increase[f'{GT_increase}-{rule_2}'][i] = [
        #                 self.compare(connectome_GT, self.draw(rule_2, matrix_shape)) for _ in range(incrs)]
        #         print(f'calculated {GT_increase}-{rule_2}')
        #     np.save(f'{score_name}_GT_increase_{matrix_shape}.npy', scores_GT_increase)

        return scores  # , scores_GT, scores_GT_increase


if __name__ == '__main__':

    network_params = {
        'mean_connection_prob': 0.1,
        'size': {
            'square': (300, 300),
            'rectangular': (450, 200)
        },
        'WS': {'p': 0.3},
        '1C': {'clusters': [(0, 50)], 'rel_cluster_weights': [10]},
        '2C': {'clusters': [(50, 85), (85, 100)], 'rel_cluster_weights': [10, 10]},
    }

    matrix_shapes = ['square', 'rectangular']
    repetitions = 100

    connectomes = Connectomes(network_params=network_params)
    os.makedirs('plots', exist_ok=True)

    for matrix_shape in ['square', 'rectangular']:
        scores = connectomes.compare_networks(matrix_shape)
        p_values, effect_sizes = connectomes.calc_statistics(scores)
        # p_values_GT, effect_sizes_GT = connectomes.calc_statistics(scores_GT)

        connectomes.plot_connectome_similarity(scores, matrix_shape)
        # connectomes.plot_p_values_reduced(p_values, scores,
        #                                   matrix_shape, savename=f'p_values_reduced_{matrix_shape}')
        # connectomes.plot_p_values_reduced(p_values_GT, scores_GT,
        #                                   matrix_shape, savename=f'p_values_reduced_{matrix_shape}_GT')
        connectomes.plot_effect_sizes(effect_sizes, scores, matrix_shape,
                                      savename=f'effect_sizes_reduced_{matrix_shape}')
        # connectomes.plot_effect_sizes(effect_sizes_GT, scores_GT, matrix_shape,
        #                               savename=f'effect_sizes_reduced_{matrix_shape}_GT')
        # connectomes.plot_p_increase(scores_GT_increase, savename=f'p_value_increase_{matrix_shape}')

    connectomes.calculate_dropoff(max_change_fraction=0.1, step_size=0.005, repetitions=10, log=True)
