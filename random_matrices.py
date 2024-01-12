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

        self.matrix_types = ['UG', 'CG', 'LRP', 'BN']

        self.titles = {
            'UG': 'uncorrelated Gauss',
            'CG': 'correlated Gauss',
            'LRP': 'low-rank perturbation',
            'BN': 'block-noise',
        }

        self.colors = {
            'UG': '#6151AC',
            'CG': '#88CCEE',
            'LRP': '#44AA99',
            'BN': '#CC6677',
        }

    def uncorr_gauss(self, dim, mean=0, var=None):
        # if var is None:
        #     var = 1 / dim
        # # mat = np.random.multivariate_normal(mean, var, (dim, dim))
        # mat = np.random.normal(mean, var, (dim, dim))
        mat = np.random.random((dim, dim))
        return mat

    def corr_gauss(self, dim, mean=0, cov_mat=None):

        # if cov_mat is None:
        #     mat = np.random.normal(0, 1, dim)
        #     cov_mat = 0.5 * (mat + mat.T) / dim

        # mean = mean * np.ones(dim)
        # mat = np.random.multivariate_normal(mean, cov_mat, dim)
        mat = np.random.random((dim, dim))
        return mat

    def low_rank_perturb(self, dim, mean=0, var=None, perturb_vec=None):

        # if var is None:
        #     var = 1 / dim
        # if perturb_vec is None:
        #     perturb_vec = np.random.bionmial(1, 0.25, dim)

        # mat = np.random.normal(mean, var, (dim, dim))
        # mat += np.outer(perturb_vec, perturb_vec)
        mat = np.random.random((dim, dim))
        return mat

    def block_with_noise(self, dim, mean=0, var=None, block_dicts=None):

        # if var is None:
        #     var = 1 / dim
        # if block_dicts is None:
        #     block_dict_1 = {
        #         'lower_row': 0,
        #         'upper_row': 30,
        #         'lower_column': 0,
        #         'upper_column': 30,
        #         'value': 0.3}
        #     block_dict_2 = {
        #         'lower_row': 60,
        #         'upper_row': 90,
        #         'lower_column': 50,
        #         'upper_column': 70,
        #         'value': -0.2}
        #     block_dicts = [block_dict_1, block_dict_2]

        # mat = np.random.normal(mean, var, (dim, dim))
        # for block_params in block_dicts:
        #     block = np.zeros((dim, dim))
        #     block[block_params['lower_row']:block_params['upper_row'],
        #           block_params['lower_column']:block_params['upper_column']] = block_params['value']
        #     mat += block
        mat = np.random.random((dim, dim))
        return mat

    # plotting

    def plot_matrix(self, matrix, name, title, fig=None, ax=None, save=True, cmap='Greens', extend='neither'):
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, ticks=[0.25, 0.75], extend=extend)
        ax.set_title(title)
        if save:
            plt.savefig(f'plots/{name}.png', dpi=600)
        return ax

    def calc_statistics(self, scores):
        # calculate p values and effect sizes between distributions
        comparisons = []
        for i, matrix_type_i in enumerate(self.matrix_types):
            for j, matrix_type_j in enumerate(self.matrix_types):
                if j >= i:
                    comparisons.append(f'{matrix_type_i}-{matrix_type_j}')
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
            if (meta_comparisons[1].split('-')[0] == meta_comparisons[1].split('-')[1]):
                if (meta_comparisons[0].split('-')[0] == meta_comparisons[0].split('-')[1]):
                    if (self.matrix_types.index(meta_comparisons[1].split('-')[0])
                            < self.matrix_types.index(meta_comparisons[0].split('-')[0])):
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
        cmap = ListedColormap(['#FFFFFF', base_color])
        return cmap

    def plot_legend(self, ax, hs, ls, loc=(0.1, 0.1), fontsize=9):
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(hs, ls, frameon=False, loc=loc, ncol=1, fontsize=fontsize)

    def plot_matrix_type_similarity(self, scores, effect_sizes, p_values):

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
        fig = plt.figure(figsize=(17, 10), layout="constrained", dpi=1200)
        matrix_type_titles = self.titles
        ax_dict = fig.subplot_mosaic(mosaic, )

        # --- PLOT CONNECTOMES ---
        ax = ax_dict['A']
        ax = self.plot_matrix(matrix=self.draw('UG'),
                                   name='UG', title=matrix_type_titles['UG'], fig=fig, ax=ax, save=False,
                                   cmap=self.colormap(self.colors['UG']))
        ax.text(-0.1, 1.15, 'A', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        ax = ax_dict['B']
        ax = self.plot_matrix(matrix=self.draw('CG'),
                                   name='square_CG', title=matrix_type_titles['CG'], fig=fig, ax=ax, save=False,
                                   cmap=self.colormap(self.colors['CG']))
        ax.text(-0.1, 1.15, 'B', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        ax = ax_dict['C']
        ax = self.plot_matrix(matrix=self.draw('LRP'),
                                   name='square_LRP', title=matrix_type_titles['LRP'], fig=fig, ax=ax,
                                   save=False, cmap=self.colormap(self.colors['LRP']))
        ax.text(-0.1, 1.15, 'C', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        ax = ax_dict['D']
        ax = self.plot_matrix(matrix=self.draw('BN'),
                                   name='square_BN', title=matrix_type_titles['BN'], fig=fig, ax=ax,
                                   save=False, cmap=self.colormap(self.colors['BN']))
        ax.text(-0.1, 1.15, 'D', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

        for matrix_type, axid in zip(self.matrix_types, ['a', 'b', 'c', 'd']):
            ax = ax_dict[axid]
            ax.text(-0.1, 1.15, axid, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

            ax.hist(scores[f'{matrix_type}-{matrix_type}'], density=True, edgecolor=self.colors[matrix_type],
                    color=self.colors[matrix_type], histtype="stepfilled", linewidth=1.5)
            for n in self.matrix_types:
                if n != matrix_type:
                    try:
                        score = scores[f'{matrix_type}-{n}']
                    except KeyError:
                        score = scores[f'{n}-{matrix_type}']
                    ax.hist(score, density=True, edgecolor=self.colors[n], histtype="step", linewidth=2.5)
            # ax.set_ylabel('Density')
            ax.set_xlabel('SAS')
            ax.set_yticks([])
            ax.spines[['right', 'top', 'left']].set_visible(False)

        for matrix_type, axid in zip(self.matrix_types, ['U', 'V', 'W', 'X']):
            ax = ax_dict[axid]
            tickcolors = self.plot_statistics(effect_sizes, p_values, ax, matrix_type)
            ax.set_xlabel(r'$\theta$')
            for lbl, c in zip(ax.get_yticklabels(), tickcolors):
                lbl.set_color(c)
        plt.savefig('plots/matrix_types_and_similarity.pdf')

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
                    _ = effect_sizes.loc[f'{matrix_type}-{matrix_type}', f'{matrix_type}-{n}']
                    comparison = f'{matrix_type}-{matrix_type}', f'{matrix_type}-{n}'
                except KeyError:
                    comparison = f'{matrix_type}-{matrix_type}', f'{n}-{matrix_type}'

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
                            changed_matrix[coordinate] = 1
                        else:
                            changed_matrix[coordinate] = 0
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
            dims=[100, 300],
            max_change_fraction=0.1, step_size=0.005, repetitions=5,
            savename='similarity_under_changes',
            log=False):

        mosaic = """
            AAABBBCCC
            """
        fig = plt.figure(figsize=(int(len(dims) * 5), 5), layout="constrained")
        ax_dict = fig.subplot_mosaic(mosaic)

        for ax_i, dim in zip(['A', 'B', 'C'], dims):
            ax = ax_dict[ax_i]
            for matrix_type in self.matrix_types:
                if matrix_type == 'UG':
                    matrix = self.uncorr_gauss(dim)
                elif matrix_type == 'CG':
                    matrix = self.corr_gauss(dim)
                elif matrix_type == 'LRP':
                    matrix = self.corr_gauss(dim)
                elif matrix_type == 'BN':
                    matrix = self.corr_gauss(dim)
                else:
                    continue
                similarity_mean, similarity_std, changes, base_matrix = self.change_matrix(
                    matrix, max_change_fraction, step_size, repetitions)
                x = changes / (np.shape(base_matrix)[0] * np.shape(base_matrix)[1]) * 100
                ax.plot(x, similarity_mean, color=self.colors[matrix_type], label=self.titles[matrix_type])
                ax.fill_between(x, similarity_mean - similarity_std, similarity_mean + similarity_std, alpha=0.3,
                                color=self.colors[matrix_type])
                ax.set_title(f'Dimension: {dim}')
            ax.set_xlabel('% of changed connections')
            ax.set_ylabel('SAS')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if log:
                # ax.set_xscale('log')
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.set_yscale('log')
            ax.set_box_aspect(1)
        if log:
            savename += '_log'
        ax.legend(frameon=False, ncol=1, fontsize=12)
        plt.savefig(f'plots/{savename}.png', dpi=600, bbox_inches='tight')

    def plot_p_increase(self, scores, savename, GT_increase='ER', increase=np.arange(3, 100, 5)):
        fig, ax = plt.subplots()
        for matrix_type in self.matrix_types:
            scores_arr = scores[f'{GT_increase}-{matrix_type}']
            mean, std = np.empty(len(scores_arr)), np.empty(len(scores_arr))
            for i, obj in enumerate(scores_arr):
                mean[i] = np.mean(obj)
                std[i] = np.std(obj)
            ax.plot(increase, mean, color=self.colors[matrix_type])
            ax.fill_between(increase, mean - std, mean + std, color=self.colors[matrix_type], alpha=0.5)
        ax.set_xlabel('Number of draws')
        ax.set_ylabel('SAS')
        plt.savefig(f'plots/{savename}.pdf')

    def draw(self, name):
        if name == 'UG':
            return self.uncorr_gauss(self.params['dim'])
        if name == 'CG':
            return self.corr_gauss(self.params['dim'])
        if name == 'LRP':
            return self.low_rank_perturb(self.params['dim'])
        if name == 'BN':
            return self.block_with_noise(self.params['dim'])

    def compare_matrix_types(self, score_name='scores', repetitions=100):

        # 100 vs 100
        try:
            scores = np.load(f'{score_name}.npy', allow_pickle=True).item()
            print('Scores found on disk. Continuing...')
        except FileNotFoundError:
            print('Scores not found on disk. Calculating...')
            scores = {}
            for rule_1, rule_2 in combinations_with_replacement(self.matrix_types, 2):
                score = [self.compare(self.draw(rule_1),
                                      self.draw(rule_2)) for _ in range(repetitions)]
                scores[f'{rule_1}-{rule_2}'] = score
                print(f"The similarity of {rule_1} and {rule_2} is {np.round(np.mean(score), 2)} ± "
                      f"{np.round(np.std(score), 2)}")
            np.save(f'{score_name}.npy', scores)

        return scores


if __name__ == '__main__':

    matrix_params = {'dim': 100}

    random_matrices = RandomMatrices(matrix_params=matrix_params)

    os.makedirs('plots', exist_ok=True)

    scores = random_matrices.compare_matrix_types(score_name='scores_random_matrices')
    p_values, effect_sizes = random_matrices.calc_statistics(scores)
    random_matrices.plot_matrix_type_similarity(scores, effect_sizes, p_values)

    # random_matrices.calculate_dropoff(max_change_fraction=0.1, step_size=0.05, repetitions=2, log=True)
