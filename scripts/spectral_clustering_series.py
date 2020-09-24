""" run a case study on composition threshold """
import os
import pathlib
import numpy as np
import pandas as pd

from VNbO2 import gp
from VNbO2 import cluster
from VNbO2.data import io
from VNbO2.utils import order_labels

def evaluate_patched_clustering(model, L, df, Nb_cutoff=0.2):
    eval_at = df['Nb'] <= Nb_cutoff
    df_eval = df.loc[:,('Nb', 'temp')].copy()
    X_clust = df.loc[sel, ('Nb', 'temp')]

    # default to high temperature cluster to extend those predictions...
    df_eval['label'] = 2
    df_eval.loc[X_clust.index, 'label'] = L
    df_eval = df_eval.loc[eval_at]

    X_eval = df_eval.loc[:,('Nb', 'temp')]
    df_eval['label'].values[:,None]
    E, error_indicator = gp.variational_log_likelihoods(X_eval.values, df_eval['label'].values[:,None], model)
    return E.numpy()

results_dir = pathlib.Path('results/spectral-clustering-series')
os.makedirs(results_dir, exist_ok=True)

# load the annotations
data = pathlib.Path('data')
annotations = pd.read_excel(data / 'Human Labels.xlsx', index_col=0)
annotations['Nb'] = 1 - annotations['V']

# load annotations from everyone!
L = annotations.iloc[:,3:8]

# merge all labels into a single table
all_labels = []
for labeler in L.columns:
    _df = annotations.loc[:,('Nb', 'temp', labeler)]
    _df = _df.rename(columns={labeler: 'label'})
    _df['labeler'] = labeler
    all_labels.append(_df)

all_labels = pd.concat(all_labels)

# fit a GP classifier to model the human expert consensus...
X = all_labels.loc[:,('Nb', 'temp')].values
y = all_labels['label'].values
m = gp.fit_sparse_gp(X, y[:,None], n_classes=3)

# load and preprocess diffraction data
# just apply linear background removal and clip the 2θ range
ar = (27, 28.5)
diffraction_data, angle, df = io.load_VNb2O3(detrend=True, angular_range=ar)

# run self-tuning spectral clustering over a range of composition croppings
Nb_thresholds = np.arange(0.10, 0.31, 0.01)

loglik = []
for idx, threshold in enumerate(Nb_thresholds):
    sel = df.Nb <= threshold
    km = cluster.self_tuning_spectral_clustering(diffraction_data[sel], metric='cosine')
    L = order_labels(km.labels_, df['Nb'][sel])
    loglik.append(evaluate_patched_clustering(m, L, df).tolist())

    results = df.loc[sel, ('Nb', 'temp')].copy()
    results['clusterlabel'] = L
    results.to_csv(results_dir / f'cluster_labels_{threshold:.02f}.csv')

df_loglik = pd.DataFrame({'Nb_threshold': Nb_thresholds, 'loglik': loglik})
df_loglik.to_csv(results_dir / 'spectral-clustering-loglik.csv')
