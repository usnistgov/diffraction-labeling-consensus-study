# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:41:20 2020

This code generates figures 3 and 4 using the file "Machine Learning Labels"

@author: jrh8
"""


import os
import gpflow
import tensorflow
import tensorflow_probability as tfp
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
#matplotlib inline


#%%
'''
Below I will generate the same set of figures, except for the Machine Learning
Labeled data sets.
'''
workdir = pathlib.Path('C://Users//jrh8//Documents//Documents//Manuscripts//2020//Fuzzy Labeling Paper//VO2-Nb Data//Diffraction Labels//Machine Learning Labels')
df = pd.read_csv(workdir / 'Compare ML Labels.csv', index_col=0)
df['Nb'] = (100 - df['V'])/100
#%%
labelers = [
    'Comp-Distance-Spectral',
    'Cosine-Local-Scaling-Spectral',
    'Cosine-Spectral',
    'VAE-Spectral'
]
labels = df.loc[:,labelers]

# get per-point distribution for each of the three phase labels
counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=1, arr=labels)

# compute the label entropy (shannon entropy) for each point
se = stats.entropy(counts, axis=1)

# visualize the label entropy
plt.scatter(df['Nb'], df['temp'], c=se, cmap='Blues', edgecolors='k');
plt.colorbar(label='label entropy')
plt.xlabel('Nb')
plt.ylabel('temperature (C)');

#%%

def fit_sparse_gp(X, lab, n_classes=2, optimize_lengthscales=True):
    """ use SVGP instead of VGP for better training scalability """

    # just use the the full combi grid for inducing points
    inducing_points = X[:152].copy()
    data=(X,lab)

    m = gpflow.models.SVGP(

        kernel=gpflow.kernels.RBF(lengthscales=[0.02,10], variance=1.0) + gpflow.kernels.White(variance=1.0),
        likelihood=gpflow.likelihoods.MultiClass(n_classes),
        inducing_variable=inducing_points,
        num_latent_gps=n_classes
    )

    # choose a reasonable prior on the GP lengthscale over composition and temperature...
    m.kernel.kernels[0].lengthscales.prior = tfp.distributions.LogNormal([np.log(0.02),np.log(10)], [1.0, 1.0])

    # m.kern.white.variance.trainable = False
    if not optimize_lengthscales:
        m.kernel.kernels[0].lengthscales.trainable = False
        m.kernel.kernels[0].variance.trainable = False

    training_loss = m.training_loss_closure(data)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(training_loss, m.trainable_variables)
    return m

#%%

all_labels = []
for labeler in labelers:
    _df = df.loc[:,('Nb', 'temp', labeler)]
    _df = _df.rename(columns={labeler: 'label'})
    _df['labeler'] = labeler
    all_labels.append(_df)

all_labels = pd.concat(all_labels)

# fit a GP classifier to model the machine learning consensus...
X = all_labels.loc[:,('Nb', 'temp')].values
y = all_labels['label'].values
#%%
m = fit_sparse_gp(X, y[:,None], n_classes=3)

#%%
mu, var = m.predict_y(X[:152])

plt.scatter(df['Nb'], df['temp'], c=tensorflow.reduce_sum(var,axis=1), cmap='Blues', edgecolors='k');
plt.colorbar(label='prediction')
plt.xlabel('Nb')
plt.ylabel('temperature (C)');
#%%
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

def snapshot(df, m):
    cmap = plt.cm.get_cmap("YlGnBu", 3)

    max_C = 0.25
    C_spacing, T_spacing = 0.001, 0.1
    grid_C, grid_T = np.meshgrid(
        np.arange(0.0, max_C+C_spacing, C_spacing),
        np.arange(df.temp.min()-50*T_spacing, df.temp.max()+50*T_spacing, T_spacing)
    )
    h, w = grid_C.shape
    extent = (np.min(grid_C), np.max(grid_C), np.min(grid_T), np.max(grid_T))
    gridpoints = np.c_[grid_C.ravel(), grid_T.ravel()]

    mu_y, var_y = m.predict_y(gridpoints)
    mu_f, var_f = m.predict_f(gridpoints)
    mu_y = mu_y.numpy()
    var_y = var_y.numpy()
    mu_f = mu_f.numpy()
    var_f = var_f.numpy()
    mu_y, var_y = mu_y.reshape(h, w, -1), var_y.reshape(h,w,-1)
    mu_f, var_f = mu_f.reshape(h, w, -1), var_f.reshape(h,w,-1)

    # color map the predictions... set alpha channel with variance...
    C = np.argmax(mu_y > 0.5, axis=-1).astype(int)
    c = cmap(C)
    a = var_y.mean(axis=-1)
    a = Normalize(a.min(), a.max(), clip=True)(a)
    c[...,-1] = 1-a

    fig, ax1 = plt.subplots(ncols=1, figsize=(6,4))
    # ax1.scatter(df['Nb'], df['temp'])
    ax1.imshow(c, origin='lower', extent=extent)
    ax1.contour(C, levels=[0.5,1.5], linestyles=['--'], colors='k', origin='lower', extent=extent)
    ax1.set_xlabel('Nb', fontsize = 20);
    ax1.set_ylabel('Temperature (C)', fontsize = 20);
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.xlim(0,0.25)
    plt.axis('auto')
    plt.tight_layout()
    return c, extent

c, extent = snapshot(df, m);

# plot the label entropy over top of the GP consensus...
plt.scatter(df['Nb'], df['temp'], c=se, cmap='Blues', edgecolors='k');
plt.savefig(workdir / "Figure 3.png")
plt.clf()

#%%
#Plot loglikelihoods from Figure 4
loglik = pd.read_csv(workdir/'cluster_assignment_loglik_all.csv', index_col=0)
loglik = loglik.sort_values(by=['Nb', 'temp']).reset_index()

ll = loglik.sum()[4:]

# assumes each model has the same number of parameters...
aic_ = -2 * ll
idx_best = np.argmin(aic_)
log_rel_lik = (aic_[idx_best] - aic_) / 2
log_rel_lik = log_rel_lik.sort_values(ascending=False)
fig, ax = plt.subplots()

# ax.plot(log_rel_lik.values)
xticks = list(map(lambda s: s.replace('_', '-'), log_rel_lik.index))
log_rel_lik.plot(marker='o')
ax.set_xticks(range(log_rel_lik.size))
ax.set_xticklabels(xticks, rotation=45, ha='right', fontsize=10)
ax.set(ylabel='log relative likelihood')
plt.savefig(workdir / 'cluster_log_relative_likelihoods_2.png', bbox_inches='tight')
plt.clf()

#%%

# run a case study on composition threshold
from VNbO2 import cluster
from VNbO2.data import io

# Figure 5: visualize spectral clustering results as the composition threshold varies
Nb_thresholds =  [0.18, 0.2, 0.22, 0.24, 0.26]

fig, axes = plt.subplots(ncols=4, figsize=(16,2.5), sharey=True)

# load and preprocess diffraction data
# just apply linear background removal and clip the 2θ range
ar = (27, 28.5)
diffraction_data, angle, df = io.load_VNb2O3(detrend=True, angular_range=ar)

for idx, (ax, threshold) in enumerate(zip(axes, Nb_thresholds)):
    sel = df.Nb <= threshold
    km = cluster.self_tuning_spectral_clustering(diffraction_data[sel], metric='cosine')
    L = order_labels(km.labels_, df['Nb'][sel])
    ax.scatter(df['Nb'][sel], df['temp'][sel], c=L)
    title = '$Nb_{max}$ = ' + f'{threshold}'
    ax.set(xlim=(0,0.205), xlabel='Nb', title=title)
    ax.annotate('abcd'[idx], xy=(0.02,1.05), xycoords='axes fraction')

axes[0].set(ylabel='T ($^\circ$ C)')
plt.savefig('figures/spectral_composition_study.png', bbox_inches='tight')
plt.show()
