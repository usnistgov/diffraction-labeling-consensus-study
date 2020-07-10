import os
import sys
import json
import yaml
import click
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import manifold

import matplotlib
import matplotlib.pyplot as plt

sys.path.append('.')
from VNbO2.data import io
from VNbO2 import kernel

composition_label = r'$\mathrm{V}_{1-x}\mathrm{W}_{x}\mathrm{O}_2$'

# specify some matplotlib settings
plot_params_path = os.path.join(os.path.dirname(__file__), 'matplotlib-params.json')
with open(plot_params_path, 'r') as f:
    matplotlib.rcParams.update(json.load(f))

def plot_results(df, labels):

    # spectral embedding drops the first eigenvalue, but not Shi&Malik spectral clustering
    col = ['m', 'c', 'y']

    for label, c in zip(label_order, col):
        plt.scatter(df['Nb'][labels == label], df['temp'][labels == label], color=c)

    plt.xlabel('composition')
    plt.ylabel('temperature')
    plt.tight_layout()
    plt.savefig('test.png')

def self_tuning_spectral_clustering(Y, metric='cosine'):

    if metric == 'cosine':
        D = kernel.pairwise.cosine_distances(Y)
    elif metric == 'hellinger':
        D = kernel.hellinger_distance(Y)
    elif metric == 'chi2':
        D = kernel.chi2_distance(Y)
    elif metric == 'js_divergence':
        D = kernel.js_divergence(Y)
    elif metric == 'alternative_cosine':
        K = kernel.locally_scaled_cosine_rbf(Y)
    else:
        raise NotImplementedError(f'{metric} is not available.')

    if metric != 'alternative_cosine':
        # compute the "self-tuning" version of the kernel matrix
        K = kernel.locally_scaled_rbf(D)

    Dg = np.diag(np.sum(K, axis=0))
    D2_inv = np.linalg.inv(np.linalg.cholesky(Dg))
    KK = D2_inv @ K @ D2_inv

    nc = 3
    # spectral embedding drops the first eigenvalue, but not Shi&Malik spectral clustering...
    # spec = manifold.SpectralEmbedding(n_components=nc, affinity='precomputed')
    # Z = spec.fit_transform(A)

    Z = manifold.spectral_embedding(KK, n_components=nc, drop_first=True)

    km = cluster.KMeans(n_clusters=3, n_init=100)
    km.fit(Z / np.linalg.norm(Z, axis=1)[:,None])

    km = cluster.SpectralClustering(n_clusters=nc, n_init=100, affinity='precomputed')
    # km.fit(1-(D/np.max(D)))
    km.fit(KK)

    return km

@click.command()
@click.argument('config-file', type=click.Path())
def run(config_file):
    """ run self-tuning spectral clustering on the VNb2O3 dataset"""

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # default configuration values...
    max_Nb = config.get('max_Nb', 0.2)
    metric = config.get('metric', 'cosine')
    detrend = config.get('detrend', True)
    post_normalize = config.get('post_normalize', False)
    angular_range = config.get('angular_range', (27, 28.5))
    constant_baseline_removal = config.get('constant_baseline_removal', False)

    # load the VNbO2 dataset with linear baseline removal
    # clip the angular range down to the range surrounding the diffraction peaks
    Y, angle, df = io.load_VNb2O3(detrend=detrend, angular_range=angular_range, post_normalize=post_normalize)

    if constant_baseline_removal:
        Y = Y - Y.min(axis=1)[:,None] + 1e-6

    # drop the label column...
    del df['Label']

    # limit the compositional range to <= 30% Nb
    sel = (df.Nb <= max_Nb)
    df = df[sel]
    Y = Y[sel]
    C = df['Nb'].values

    print(f'running {metric} self-tuning spectral clustering')
    km = self_tuning_spectral_clustering(Y, metric=metric)

    # reorder labels...
    C_avg = []
    for label in np.unique(km.labels_):
        C_avg.append(np.mean(C[km.labels_==label]))

    label_order = np.argsort(C_avg)[::-1]
    labels = label_order[km.labels_]

    df['cluster_label'] = labels
    results_file = os.path.join(os.path.dirname(config_file), f'results-{metric}.csv')
    df.to_csv(results_file)

if __name__ ==  '__main__':
    run()
