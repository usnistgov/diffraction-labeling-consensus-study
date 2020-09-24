import glob
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results_dir = pathlib.Path('results/spectral-clustering-series')

# figure 5: examples of cluster results
fig, axes = plt.subplots(ncols=4, figsize=(16,2.5), sharey=True)

thresholds = [0.18, 0.2, 0.22, 0.24, 0.26]
for idx, (ax, threshold) in enumerate(zip(axes, thresholds)):
    df = pd.read_csv(results_dir / f'cluster_labels_{threshold:.02f}.csv', index_col=0)

    ax.scatter(df['Nb'], df['temp'], c=df['clusterlabel'])
    title = '$Nb_{max}$ = ' + f'{threshold}'
    ax.set(xlim=(0,0.205), xlabel='Nb', title=title)
    ax.annotate('abcd'[idx], xy=(0.02,1.05), xycoords='axes fraction')

axes[0].set(ylabel='T ($^\circ$ C)')
plt.savefig('figures/spectral_composition_study.png', bbox_inches='tight')
plt.clf()
plt.close()

## Figure 6: relative log likelihoods
plt.figure()
df = pd.read_csv(results_dir / 'spectral-clustering-loglik.csv', index_col=0)

# caution -- `eval`ing the contents of some random file is a bad idea....
# probably this should get rewritten to avoid parsing a list from a string in a dataframe column...

N = len(eval(df['loglik'][0]))
aic_ = -2 * np.array([sum(eval(ll)) for ll in df['loglik']])
idx_best = np.argmin(aic_)
Nb_best = df['Nb_threshold'].iloc[idx_best]

log_rel_lik = (aic_[idx_best] - aic_) / 2
plt.plot(df['Nb_threshold'], log_rel_lik, marker='o')
plt.xlabel('$Nb_{max}$ threshold')
plt.ylabel('log relative likelihood')
plt.axvline(Nb_best, linestyle='--', alpha=0.5)
plt.axhline(0.0, linestyle='--', color='k')
plt.xlim(0.1, 0.3)
plt.savefig('figures/spectral_composition_study_rel_lik.png', bbox_inches='tight')
