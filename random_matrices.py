import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from itertools import product, combinations, combinations_with_replacement
from singular_angles import SingularAngles
import random
from scipy import stats
import xarray as xr
import os

# set fonttype so Avenir can be used with pdf format
# mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['xtick.labelsize'] = 13  # Adjust x-axis tick label size
plt.rcParams['ytick.labelsize'] = 13  # Adjust y-axis tick label size
plt.rcParams['axes.labelsize'] = 15  # Adjust x and y-axis label size


class RandomMatrices(SingularAngles):
    """docstring for RandomMatrices"""

    def __init__(self, matrix_params):
        super(RandomMatrices, self).__init__()

        self.params = matrix_params

        self.matrix_types = ['UN', 'CN', 'UN-LR', 'UN-B']
        self.matrix_shapes = ['square', 'rectangular']

        self.titles = {
            'UN': 'Uncorrelated normal',
            'CN': 'Correlated normal',
            'UN-LR': 'Uncorrelated normal\nwith low-rank perturbations',
            'UN-B': 'Uncorrelated normal\nwith blocks',
        }

        self.labels = {
            'UN': 'UN',
            'CN': 'CN',
            'UN-LR': 'UN-LR',
            'UN-B': 'UN-B',
        }

        self.network_labels = {
            'UN': 'UN',
            'CN': 'CN',
            'UN-LR': 'UN-LR',
            'UN-B': 'UN-B',
        }

        self.colors = {
            'UN': '#8C8C96',
            'CN': '#1E6429',
            'UN-LR': '#86670F',
            'UN-B': '#156177',
        }

        self.perturb_vec = {}

        self.network_colors = {
            'ER': '#6151AC',
            'DCM': '#88CCEE',
            '1C': '#44AA99',
            '2C': '#CC6677',
            '2C_diff': '#D0A5AC',
            'WS': '#DDCC77',
            'BA': '#EE8866',
        }

        self.network_labels = {
            'ER': 'ER',
            'DCM': 'DCM',
            '1C': 'OC',
            '2C': 'TC',
            'WS': 'WS',
            'BA': 'BA',
        }

        self.network_titles = {
            'ER': 'Erdős-Rényi',
            'DCM': 'Directed configuration model',
            '1C': 'One cluster',
            '2C': 'Two clusters',
            'WS': 'Watts-Strogatz',
            'BA': 'Barabasi-Albert',
        }

    def uncorr_gauss(self, size, mean=0, var=None):
        if var is None:
            var = 1 / np.sqrt(size[0] * size[1])
        mat = np.random.normal(mean, var, size)
        return mat

    def corr_gauss(self, size, mean=None, cov_mat=None):

        if mean is None:
            mean_0 = np.zeros(size[0])
            mean_1 = np.zeros(size[1])
        if cov_mat is None:
            omega = 20
            axis_0 = np.linspace(-size[0] / 2, size[0] / 2, size[0])
            axis_1 = np.linspace(-size[1] / 2, size[1] / 2, size[1])
            cov_mat_0 = (np.exp(-np.abs(axis_0[:, np.newaxis] - axis_0[np.newaxis, :]) / size[0])
                         * np.cos(np.abs(axis_0[:, np.newaxis] - axis_0[np.newaxis, :]) * omega / size[0]) / size[0]**(1.85))
            cov_mat_1 = (np.exp(-np.abs(axis_1[:, np.newaxis] - axis_1[np.newaxis, :]) / size[1])
                         * np.cos(np.abs(axis_1[:, np.newaxis] - axis_1[np.newaxis, :]) * omega / size[1]) / size[1]**(1.85))
        mat_0 = np.random.multivariate_normal(mean_0, cov_mat_0, size[1]) # -> shap (size[1], size[0])
        mat_1 = np.random.multivariate_normal(mean_1, cov_mat_1, size[0]) # -> shap (size[0], size[1])
        mat = (mat_0.T + mat_1) / 2

        return mat

    def low_rank_perturb(self, size, mean=0, var=None):

        if var is None:
            var = 1 / np.sqrt(size[0] * size[1])

        mat = np.random.normal(mean, var, size)

        try:
            mat += np.outer(self.perturb_vec[size][:size[0]], self.perturb_vec[size][:size[1]])
        except KeyError:
            self.perturb_vec[size] = np.random.binomial(1, 0.5, np.max(size)) / np.sqrt(np.max(size))
            mat += np.outer(self.perturb_vec[size][:size[0]], self.perturb_vec[size][:size[1]])

        return mat

    def block_with_noise(self, size, mean=0, var=None, block_dicts=None):

        if var is None:
            var = 1 / np.sqrt(size[0] * size[1])
        if block_dicts is None:
            block_dict_1 = {
                'lower_row': 0,
                'upper_row': 30,
                'lower_column': 0,
                'upper_column': 30,
                'value': var}
            block_dict_2 = {
                'lower_row': 80,
                'upper_row': 100,
                'lower_column': 40,
                'upper_column': 60,
                'value': -var}
            block_dicts = [block_dict_1, block_dict_2]

        mat = np.random.normal(mean, var, size)
        for block_params in block_dicts:
            block = np.zeros(size)
            block[block_params['lower_row']:block_params['upper_row'],
                  block_params['lower_column']:block_params['upper_column']] = block_params['value']
            mat += block

        return mat

    # plotting

    def plot_matrix(self, matrix, title, fig=None, ax=None, cmap='Greens', extend='neither'):
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap=cmap, norm=plt.Normalize(-np.max(np.abs(matrix)), np.max(np.abs(matrix))))
        cbar = fig.colorbar(im, ax=ax, extend=extend)
        cbar.set_ticks([-0.01, 0, 0.01])
        ax.set_title(title)
        if np.shape(matrix)[0] == np.shape(matrix)[1]:
            ax.set_xticks([0, 100, 200, 300])
            ax.set_yticks([0, 100, 200, 300])
        else:
            ax.set_xticks([0, 100, 200])
            ax.set_yticks([0, 100, 200, 300, 400])
        return ax

    def calc_statistics(self, scores):
        # calculate p values and effect sizes between distributions
        comparisons = []
        for i, matrix_type_i in enumerate(self.matrix_types):
            for j, matrix_type_j in enumerate(self.matrix_types):
                if j >= i:
                    comparisons.append(f'{matrix_type_i}_{matrix_type_j}')
        dummy_data_array = np.zeros((len(comparisons), len(comparisons)))
        p_values = xr.DataArray(
            dummy_data_array,
            coords={'comparison_0': comparisons, 'comparison_1': comparisons},
            dims=['comparison_0', 'comparison_1'])
        effect_sizes = xr.DataArray(
            dummy_data_array.copy(),
            coords={'comparison_0': comparisons, 'comparison_1': comparisons},
            dims=['comparison_0', 'comparison_1'])

        for meta_comparisons in combinations(comparisons, 2):

            # if self-similarity is at position 1, swap positions
            if (meta_comparisons[1].split('_')[0] == meta_comparisons[1].split('_')[1]):
                if (meta_comparisons[0].split('_')[0] == meta_comparisons[0].split('_')[1]):
                    if (self.matrix_types.index(meta_comparisons[1].split('_')[0])
                            < self.matrix_types.index(meta_comparisons[0].split('_')[0])):
                        meta_comparisons = tuple(reversed(meta_comparisons))
                else:
                    meta_comparisons = tuple(reversed(meta_comparisons))

            if ((meta_comparisons[0].split('_')[0] == meta_comparisons[0].split('_')[1])
                    and (meta_comparisons[0].split('_')[0] in meta_comparisons[1].split('_'))):
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
        cmap = ListedColormap(['#FFFFFF', base_color])
        return cmap

    def plot_legend(self, ax, hs, ls, loc=(0.1, 0.1), fontsize=9):
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(hs, ls, frameon=False, loc=loc, ncol=1, fontsize=fontsize)

    def plot_matrix_type_similarity(self, scores, effect_sizes, p_values, matrix_shape):

        mosaic = """
            AAAAAaaaaaU.BBBBBbbbbbV.
            AAAAAaaaaaU.BBBBBbbbbbV.
            AAAAAaaaaaU.BBBBBbbbbbV.
            AAAAAaaaaaU.BBBBBbbbbbV.
            AAAAAaaaaaU.BBBBBbbbbbV.
            CCCCCcccccW.DDDDDdddddX.
            CCCCCcccccW.DDDDDdddddX.
            CCCCCcccccW.DDDDDdddddX.
            CCCCCcccccW.DDDDDdddddX.
            CCCCCcccccW.DDDDDdddddX.
            """
        if matrix_shape == 'square':
            fig = plt.figure(figsize=(17, 7), layout="constrained", dpi=1200)
            matrix_type_titles = self.titles
        elif matrix_shape == 'rectangular':
            fig = plt.figure(figsize=(15, 7), layout="constrained", dpi=1200)
            matrix_type_titles = self.labels

        ax_dict = fig.subplot_mosaic(mosaic, )

        # --- PLOT CONNECTOMES ---
        ax = ax_dict['A']
        ax = self.plot_matrix(matrix=self.draw('UN', self.params['size'][matrix_shape]),
                              title=matrix_type_titles['UN'], fig=fig, ax=ax,
                              cmap=self.colormap(self.colors['UN']))
        ax.text(-0.1, 1.15, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        ax = ax_dict['B']
        ax = self.plot_matrix(matrix=self.draw('CN', self.params['size'][matrix_shape]),
                              title=matrix_type_titles['CN'], fig=fig, ax=ax,
                              cmap=self.colormap(self.colors['CN']))
        ax.text(-0.1, 1.15, 'B', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        ax = ax_dict['C']
        ax = self.plot_matrix(matrix=self.draw('UN-LR', self.params['size'][matrix_shape]),
                              title=matrix_type_titles['UN-LR'], fig=fig, ax=ax,
                              cmap=self.colormap(self.colors['UN-LR']))
        ax.text(-0.1, 1.15, 'C', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        ax = ax_dict['D']
        ax = self.plot_matrix(matrix=self.draw('UN-B', self.params['size'][matrix_shape]),
                              title=matrix_type_titles['UN-B'], fig=fig, ax=ax,
                              cmap=self.colormap(self.colors['UN-B']))
        ax.text(-0.1, 1.15, 'D', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        for matrix_type, axid in zip(self.matrix_types, ['a', 'b', 'c', 'd']):
            ax = ax_dict[axid]
            ax.text(-0.1, 1.15, axid, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

            ax.hist(scores[f'{matrix_type}_{matrix_type}'], density=True, edgecolor=self.colors[matrix_type],
                    color=self.colors[matrix_type], histtype="stepfilled", linewidth=1.5)
            for n in self.matrix_types:
                if n != matrix_type:
                    try:
                        score = scores[f'{matrix_type}_{n}']
                    except KeyError:
                        score = scores[f'{n}_{matrix_type}']
                    ax.hist(score, density=True, edgecolor=self.colors[n], histtype="step", linewidth=2.5)
            # ax.set_ylabel('Density')
            ax.set_xlabel('SAS')
            ax.set_yticks([])
            ax.spines[['right', 'top', 'left']].set_visible(False)

            if matrix_type == 'UN':
                ax.set_xticks([0.020, 0.025])

        for matrix_type, axid in zip(self.matrix_types, ['U', 'V', 'W', 'X']):
            ax = ax_dict[axid]
            tickcolors = self.plot_statistics(effect_sizes, p_values, ax, matrix_type)
            ax.set_xlabel(r'$\theta$')
            for lbl, c in zip(ax.get_yticklabels(), tickcolors):
                lbl.set_color(c)
        plt.savefig(f'plots/matrices_and_similarity_{matrix_shape}.pdf')

    def plot_statistics(self, effect_sizes, p_values, ax, matrix_type):

        bars = []
        p_vals = []
        lbl = []
        tickcolors = []
        i = 0
        # Create bar plot
        for n in self.matrix_types[::-1]:
            if n != matrix_type:
                try:
                    _ = effect_sizes.loc[f'{matrix_type}_{matrix_type}', f'{matrix_type}_{n}']
                    comparison = f'{matrix_type}_{matrix_type}', f'{matrix_type}_{n}'
                except KeyError:
                    comparison = f'{matrix_type}_{matrix_type}', f'{n}_{matrix_type}'

                bars.append(ax.barh(i, effect_sizes.loc[comparison], color=self.colors[n]))
                p_vals.append(p_values.loc[comparison])
                lbl.append(n)
                tickcolors.append(self.colors[n])
                i += 1

        # Set x-axis to logarithmic scale
        # ax.set_xscale('log')
        ax.set_xlim(-1, 5)
        ax.set_xticks([1, 5])
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.set_yticks(np.arange(len(self.matrix_types) - 1))
        ax.set_yticklabels(lbl)

        # Add vertical line at x=1
        ax.axvline(x=0, color='black', lw=0.8)
        ax.axvline(x=1, color='black', ls='--', lw=0.8)

        # Iterate over bars and p_values
        for bar, p_value in zip(bars, p_vals):
            # Check if p_value is larger or smaller than 0.05
            if 0.01 < p_value < 0.05:
                # Add star to the right of the bar
                ax.annotate('*', xy=(5, bar[0].get_y() + bar[0].get_height() / 2), xytext=(3, 0),
                            textcoords='offset points', ha='left', va='center')
            elif 0.001 < p_value < 0.01:
                # Add star to the right of the bar
                ax.annotate('**', xy=(5, bar[0].get_y() + bar[0].get_height() / 2), xytext=(3, 0),
                            textcoords='offset points', ha='left', va='center')
            elif p_value < 0.001:
                # Add star to the right of the bar
                ax.annotate('***', xy=(5, bar[0].get_y() + bar[0].get_height() / 2), xytext=(3, 0),
                            textcoords='offset points', ha='left', va='center')
            else:
                # Add 'n.s.' string to the right of the bar
                ax.annotate('n.s.', xy=(5, bar[0].get_y() + bar[0].get_height() / 2), xytext=(3, 0),
                            textcoords='offset points', ha='left', va='center')

        return tickcolors

    # --- CALCULATE SIMILARITY FOR INCREASINGLY DIFFERENT MATRICES

    def change_matrix(self, base_matrix, change_range):

        scores = np.zeros(len(change_range))
        for i, change in enumerate(change_range):
            changed_matrix = base_matrix.copy()
            if change > 0:
                changed_matrix += np.random.normal(0, change / np.shape(changed_matrix)[0]**3, np.shape(changed_matrix))
            scores[i] = self.compare(base_matrix, changed_matrix)
        return scores

    def characterization(self, dim_range=np.linspace(100, 500, 21), max_change_fraction=0.1, step_size=0.005,
                         repetitions=10, savename='characterization', log=False):

        mosaic = """
            AAABBB
            CCCDDD
            """
        fig = plt.figure(figsize=(9.5, 9.5), layout="constrained")
        ax = fig.subplot_mosaic(mosaic)

        # increase dimension
        try:
            dim_scores = xr.load_dataarray('increase_dim_scores.nc').load()
        except FileNotFoundError:
            dim_scores = xr.DataArray(
                np.zeros((len(self.matrix_types), len(dim_range), repetitions)),
                coords={'matrix_type': self.matrix_types, 'dim': dim_range, 'repetition': np.arange(repetitions)},
                dims=['matrix_type', 'dim', 'repetition'])
            for dim in dim_range:
                for matrix_type in self.matrix_types:
                    dim_scores.loc[matrix_type, int(dim), :] = [self.compare(
                        self.draw(matrix_type, (int(dim), int(dim))), self.draw(matrix_type, (int(dim), int(dim))))
                        for _ in range(repetitions)]
            dim_scores.to_netcdf('increase_dim_scores.nc')

        for matrix_type in self.matrix_types:
            mean = np.mean(dim_scores.loc[matrix_type], axis=-1)
            std = np.std(dim_scores.loc[matrix_type], axis=-1)
            ax['A'].plot(dim_range, mean, label=matrix_type, color=self.colors[matrix_type])
            ax['A'].fill_between(dim_range, mean - std, mean + std, alpha=0.3, color=self.colors[matrix_type], lw=0)
        ax['A'].set_xlabel(r'Dimensionality $N$')
        ax['A'].set_ylabel('SAS')
        ax['A'].spines['top'].set_visible(False)
        ax['A'].spines['right'].set_visible(False)
        ax['A'].text(-0.1, 1.15, 'A', transform=ax['A'].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        # increase number of changed connections
        size = self.params['size']['square'][0]**2
        max_changes = int(size * max_change_fraction)
        change_range = np.insert(np.arange(max_changes + 1, step=size * step_size).astype(int),
                                 1, np.arange(1, 100, step=10))
        try:
            change_scores = xr.load_dataarray('increase_change_scores.nc').load()
        except FileNotFoundError:
            change_scores = xr.DataArray(
                np.zeros((len(self.matrix_types), len(change_range), repetitions)),
                coords={'matrix_type': self.matrix_types, 'change': change_range, 'repetition': np.arange(repetitions)},
                dims=['matrix_type', 'change', 'repetition'])
            for matrix_type in self.matrix_types:
                for repetition in range(repetitions):
                    base_matrix = self.draw(matrix_type)
                    change_scores.loc[matrix_type, :, repetition] = self.change_matrix(base_matrix, change_range)
            change_scores.to_netcdf('increase_change_scores.nc')

        x = change_range / self.params['size']['square'][0]**2

        for matrix_type in self.matrix_types:
            mean = np.mean(change_scores.loc[matrix_type], axis=-1)
            std = np.std(change_scores.loc[matrix_type], axis=-1)
            ax['B'].plot(x, mean, color=self.colors[matrix_type], label=self.titles[matrix_type])
            ax['B'].fill_between(x, mean - std, mean + std, alpha=0.3, color=self.colors[matrix_type], lw=0)
        ax['B'].set_xlabel(r'relative variance of perturbation $f$')
        ax['B'].set_ylabel('SAS')
        ax['B'].spines['top'].set_visible(False)
        ax['B'].spines['right'].set_visible(False)
        # if log:
        #     ax['B'].yaxis.set_major_formatter(ticker.ScalarFormatter())
        #     ax['B'].set_yscale('log')
        ax['B'].legend(frameon=False, ncol=1, fontsize=12, loc='upper right')
        ax['B'].text(-0.1, 1.15, 'B', transform=ax['B'].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        # ---- data from network models ----
        dim_scores = xr.load_dataarray('increase_dim_scores_networks.nc').load()
        change_scores = xr.load_dataarray('increase_change_scores_networks.nc').load()

        for network in dim_scores.coords['network']:
            network = str(network.values)
            mean = np.mean(dim_scores.loc[network], axis=-1).values
            std = np.std(dim_scores.loc[network], axis=-1).values
            ax['C'].plot(dim_range, mean, label=self.network_labels[network], color=self.network_colors[network])
            ax['C'].fill_between(dim_range, mean - std, mean + std, alpha=0.3, color=self.network_colors[network], lw=0)
        ax['C'].set_xlabel(r'Network size $N$')
        ax['C'].set_ylabel('SAS')
        ax['C'].spines['top'].set_visible(False)
        ax['C'].spines['right'].set_visible(False)
        ax['C'].text(-0.1, 1.15, 'C', transform=ax['C'].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        for network in change_scores.coords['network']:
            network = str(network.values)
            mean = np.mean(change_scores.loc[network], axis=-1)
            std = np.std(change_scores.loc[network], axis=-1)
            ax['D'].plot(change_scores.coords['change'].values, mean,
                         color=self.network_colors[network], label=self.network_titles[network])
            ax['D'].fill_between(change_scores.coords['change'].values, mean - std, mean +
                                 std, alpha=0.3, color=self.network_colors[network], lw=0)
        ax['D'].set_xlabel(r'fraction of changed connections $g$ [%]')
        ax['D'].set_ylabel('SAS')
        ax['D'].spines['top'].set_visible(False)
        ax['D'].spines['right'].set_visible(False)
        if log:
            ax['D'].yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax['D'].set_yscale('log')
        ax['D'].legend(frameon=False, ncol=1, fontsize=12, loc='upper right')
        ax['D'].text(-0.1, 1.15, 'D', transform=ax['D'].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        plt.savefig(f'plots/{savename}.pdf', bbox_inches='tight')

    def draw(self, name, size=None, matrix_shape='square'):
        if size is None:
            size = self.params['size'][matrix_shape]
        if name == 'UN':
            return self.uncorr_gauss(size)
        elif name == 'CN':
            return self.corr_gauss(size)
        elif name == 'UN-LR':
            return self.low_rank_perturb(size)
        elif name == 'UN-B':
            return self.block_with_noise(size)

    def compare_matrix_types(self, matrix_shape, score_name='scores', repetitions=100):

        # 100 vs 100
        try:
            scores = np.load(f'{score_name}_{matrix_shape}.npy', allow_pickle=True).item()
            print('Scores found on disk. Continuing...')
        except FileNotFoundError:
            print('Scores not found on disk. Calculating...')
            scores = {}
            for rule_1, rule_2 in combinations_with_replacement(self.matrix_types, 2):
                score = [self.compare(self.draw(rule_1, matrix_shape=matrix_shape),
                                      self.draw(rule_2, matrix_shape=matrix_shape)) for _ in range(repetitions)]
                scores[f'{rule_1}_{rule_2}'] = score
                print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(score), 2)} ± "
                      f"{np.round(np.std(score), 2)}")
            np.save(f'{score_name}_{matrix_shape}.npy', scores)

        return scores


if __name__ == '__main__':

    matrix_params = {'size': {'square': (300, 300), 'rectangular': (450, 200)}}
    matrix_shapes = ['rectangular']

    random_matrices = RandomMatrices(matrix_params=matrix_params)

    os.makedirs('plots', exist_ok=True)

    for matrix_shape in matrix_shapes:
        scores = random_matrices.compare_matrix_types(
            matrix_shape, score_name='scores_random_matrices', repetitions=100)
        p_values, effect_sizes = random_matrices.calc_statistics(scores)
        random_matrices.plot_matrix_type_similarity(scores, effect_sizes, p_values, matrix_shape)

    # random_matrices.characterization(max_change_fraction=1.0, step_size=0.05, repetitions=5, log=True)
