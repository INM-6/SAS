from random_matrices import RandomMatrices
from connectomes import Connectomes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr

# set fonttype so Avenir can be used with pdf format
# mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['xtick.labelsize'] = 13  # Adjust x-axis tick label size
plt.rcParams['ytick.labelsize'] = 13  # Adjust y-axis tick label size
plt.rcParams['axes.labelsize'] = 15  # Adjust x and y-axis label size

# instantiate classes
connectomes = Connectomes()
random_matrices = RandomMatrices()

mosaic = """
    AAABBB
    CCCDDD
    """
fig = plt.figure(figsize=(9.5, 9.5), layout="constrained")
ax = fig.subplot_mosaic(mosaic)

# ---- data from random matrices ----

dim_scores = xr.load_dataarray('results/increase_dim_scores_random_matrices.nc').load()
change_scores = xr.load_dataarray('results/increase_change_scores_random_matrices.nc').load()
dim_range = dim_scores.coords['dim'].values

for matrix_type in random_matrices.matrix_types:
    mean = np.mean(dim_scores.loc[matrix_type], axis=-1)
    std = np.std(dim_scores.loc[matrix_type], axis=-1)
    ax['A'].plot(dim_range, mean, label=matrix_type, color=random_matrices.colors[matrix_type])
    ax['A'].fill_between(dim_range, mean - std, mean + std, alpha=0.3, color=random_matrices.colors[matrix_type], lw=0)
ax['A'].set_xlabel(r'Dimensionality $N$')
ax['A'].set_ylabel('SAS')
ax['A'].spines['top'].set_visible(False)
ax['A'].spines['right'].set_visible(False)
ax['A'].text(-0.1, 1.15, 'A', transform=ax['A'].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

change = change_scores.coords['change'].values / random_matrices.params['size']['square'][0]**2

for matrix_type in random_matrices.matrix_types:
    mean = np.mean(change_scores.loc[matrix_type], axis=-1)
    std = np.std(change_scores.loc[matrix_type], axis=-1)
    ax['B'].plot(change, mean, color=random_matrices.colors[matrix_type], label=random_matrices.titles[matrix_type])
    ax['B'].fill_between(change, mean - std, mean + std, alpha=0.3, color=random_matrices.colors[matrix_type], lw=0)
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
dim_scores = xr.load_dataarray('results/increase_dim_scores_connectomes.nc').load()
change_scores = xr.load_dataarray('results/increase_change_scores_connectomes.nc').load()
dim_range = dim_scores.coords['dim'].values

for network in dim_scores.coords['network']:
    network = str(network.values)
    mean = np.mean(dim_scores.loc[network], axis=-1).values
    std = np.std(dim_scores.loc[network], axis=-1).values
    ax['C'].plot(dim_range, mean, label=connectomes.labels[network], color=connectomes.colors[network])
    ax['C'].fill_between(dim_range, mean - std, mean + std, alpha=0.3, color=connectomes.colors[network], lw=0)
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
                 color=connectomes.colors[network], label=connectomes.titles[network])
    ax['D'].fill_between(change_scores.coords['change'].values, mean - std, mean +
                         std, alpha=0.3, color=connectomes.colors[network], lw=0)
ax['D'].set_xlabel(r'fraction of changed connections $g$ [%]')
ax['D'].set_ylabel('SAS')
ax['D'].spines['top'].set_visible(False)
ax['D'].spines['right'].set_visible(False)
ax['D'].yaxis.set_major_formatter(ticker.ScalarFormatter())
ax['D'].set_yscale('log')
ax['D'].legend(frameon=False, ncol=1, fontsize=12, loc='upper right')
ax['D'].text(-0.1, 1.15, 'D', transform=ax['D'].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

plt.savefig('plots/characterization.pdf', bbox_inches='tight')
