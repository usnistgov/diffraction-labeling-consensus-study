import os
import glob
import json
import numpy as np
import pandas as pd

from . import preprocess

data_root = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data')

def load_VNb2O3(detrend=False, smooth=False, angular_range=None, post_normalize=False, ord=1):
    """ load VNb2O3 metadata into pandas, and XRD data into numpy (via pandas)
    Combiview metadata comes in multiple files
    join on the same index across all these files...
    """

    metadata = [
        # 'VO2 - Nb2O3 Composition Combiview.txt',
        'VO2 - Nb2O3 Phase Labels Combiview.txt',
        'VO2 - Nb2O3 Composition and temp Combiview.txt',
        'VO2 - Nb2O3 Position Combiview.txt'
    ]
    metadata = [os.path.join(data_root, filename) for filename in metadata]

    df = pd.concat(
        list(map(lambda f: pd.read_csv(f, sep='\t'), metadata)),
        axis=1
    )

    # drop duplicate composition column...
    df = df.loc[:,~df.columns.duplicated()]

    df.rename(columns={'Unnamed: 3': 'sample_idx'})

    xrd = pd.read_csv(os.path.join(data_root, 'VO2 -Nb2O3 XRD Combiview.txt'), sep='\t')
    X = np.array(xrd)
    angle = np.array(xrd.keys(), dtype=float)

    # use zero-indexed labels...
    df['Label'] = df['Label'] - 1

    # apply pre-processing
    if angular_range is not None:
        angle, X = preprocess.trim_angular_range(angle, X, angular_range=(18,37.2))

    if detrend:
        X = np.array([
            preprocess.detrend_pattern(angle, x)
            for x in X
        ])

    if smooth:
        X = preprocess.smooth_patterns(X)

    angle, X = preprocess.trim_angular_range(angle, X, angular_range=angular_range)

    if post_normalize:
        X = X / np.linalg.norm(X, ord=ord, axis=-1)[:,None]


    return X, angle, df

def read_pattern(datafile, detrend=False, smooth=False, pretrim=True, angular_range=None, post_normalize=False, ord=1, fmt='xy'):
    """ parse an xy file """

    if fmt == 'xy':
        skiprows = 1
    elif fmt == 'plt':
        skiprows = 16

    df = pd.read_csv(datafile, skiprows=skiprows, header=None, delimiter=' ', names=['angle', 'energy'])
    angle, X = np.array(df.angle), np.array(df.energy)

    # apply pre-processing
    if pretrim:
        angle, X = preprocess.trim_angular_range(angle, X, angular_range=(18,37.2))

    if detrend:
        X = preprocess.detrend_pattern(angle, X)

    if smooth:
        X = preprocess.smooth_patterns(X)

    angle, X = preprocess.trim_angular_range(angle, X, angular_range=angular_range)

    if post_normalize:
        X = X / np.linalg.norm(X, ord=ord)

    return angle, X
