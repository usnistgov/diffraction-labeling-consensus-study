import numpy as np
from scipy import signal
from sklearn import linear_model

def select_angular_range(angle, angular_range=(26, 30), tol=1e-4):
    min_angle, max_angle = angular_range

    id_angle = (angle - min_angle >= -tol) & (angle - max_angle <= tol)

    return id_angle

def trim_angular_range(angle, pattern, angular_range=(26, 30), tol=1e-4):

    if angular_range is None:
        return angle, pattern

    id_angle = select_angular_range(angle, angular_range=angular_range, tol=tol)

    return angle[id_angle], pattern[...,id_angle]

def detrend_pattern(angle, pattern, support=None, return_bias=False):

    reg = linear_model.BayesianRidge()

    if support is not None:
        reg.fit(angle[support,None], pattern[support])
    else:
        mean_intensity = np.mean(np.abs(pattern))
        std_intensity = np.std(np.abs(pattern))
        support = np.abs(pattern) < mean_intensity + 2*std_intensity

        reg.fit(angle[support,None], pattern[support])

    bias = reg.predict(angle[:,None])

    if return_bias:
        return pattern - bias, bias
    else:
        return pattern - bias

def detrend_dataset(angle, X):
    XX = []
    for x in X:
        mean_intensity = np.mean(np.abs(x))
        std_intensity = np.std(np.abs(x))
        support = np.abs(x) < mean_intensity + 2*std_intensity
        pattern, bias = detrend_pattern(angle, x, support=support, return_bias=True)
        XX.append(pattern)
    return np.array(XX)

def smooth_patterns(patterns, order=2, Wn=0.025, clip=True):
    """ apply a lowpass filter to suppress noise, with optional clipping to zero """

    b, a = signal.butter(order, Wn, analog=False)
    XXclip = signal.filtfilt(b, a, patterns, axis=-1)

    if clip:
        XXclip = np.clip(XXclip, 0.0, 9999)

    return XXclip
