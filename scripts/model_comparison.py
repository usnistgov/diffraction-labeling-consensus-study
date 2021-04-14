import re
import os
import pathlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import sys

sys.path.append(".")

from VNbO2.data import io
from VNbO2.utils import order_labels

# The annotations and non-VAE results have been curated manually.
# Just load them from excel and csv files:
data = pathlib.Path("data")
annotations = pd.read_excel(data / "Human Labels.xlsx", index_col=0)
annotations["Nb"] = 1 - annotations["V"]

algo_labels = pd.read_csv(data / "Compare ML Labels.csv", index_col=0)
algo_labels["V"] *= 0.01
algo_labels["Nb"] *= 0.01

# load the dataset:
diffraction_data, angle, df = io.load_VNb2O3(detrend=False)


# human labeler columns start with HL
labelers = list(filter(lambda s: re.match("HL", s), annotations.columns))
L = annotations.loc[:, labelers]
annotations.loc[:, labelers]

# drop HL3
L.iloc[:, 1:]

# get per-point distribution for each of the three phase labels
counts = np.apply_along_axis(
    lambda x: np.bincount(x, minlength=3), axis=1, arr=L.iloc[:, 1:]
)

# compute the label entropy (shannon entropy) for each point
se = stats.entropy(counts, axis=1)

# We can use this consensus to rank cluster assignments
# relative to the human annotators.

weighted_scores = {}
acc = {}
# human labeler columns start with HL
labelers = list(filter(lambda s: re.match("HL", s), annotations.columns))

# with four annotators and three categories
# the max possible entropy is:
entropy_max = stats.entropy([1, 1, 1])

L = annotations.loc[:, labelers]
L = L[annotations.Nb <= algo_labels.Nb.max()]

# drop HL3...
L = L.iloc[:, 1:].values

label_entropy = se[annotations.Nb <= algo_labels.Nb.max()]
weights = 1 - label_entropy / entropy_max

# consensus labels: take the mode among experts
mode, count = stats.mode(L, axis=1)
mode = np.squeeze(mode)

# enforce mixed-phase for ties
mode[label_entropy == stats.entropy([0, 2, 2])] = 1

algo_keys = algo_labels.columns[3:]
X = algo_labels.loc[:, ("Nb", "temp")].values.astype(np.float64)
loglik = algo_labels.iloc[:, :3].copy()

for algo_key in algo_keys:
    Y = order_labels(algo_labels[algo_key], algo_labels["Nb"])
    E = weights * (Y == mode)

    acc[algo_key] = np.mean(Y == mode)
    weighted_scores[algo_key] = np.mean(E)
    print(algo_key, np.mean(E))

# plot weighted model scores
fig, ax = plt.subplots()

# sort by decreasing weighted score
order = np.argsort(list(weighted_scores.values()))[::-1]

score = np.array(list(weighted_scores.values()))[order]
_labels = np.array(list(weighted_scores.keys()))[order]
acc_score = [acc[key] for key in _labels]

plt.plot(score, marker="o", label="SE weighted accuracy")
plt.plot(acc_score, marker="s", label="accuracy")

ax.set_xticks(range(len(score)))
_labels
xticks = list(map(lambda s: s.replace("-", " "), _labels))
xticks = list(map(lambda s: s.replace(" Spectral", ""), xticks))

format = {
    "Cosine Local Scaling": "Cosine\nLocal Scaling",
    "Comp Distance": "Comp\nDistance",
}

xticks = [format[xt] if xt in format else xt for xt in xticks]
xticks
# ax.set_xticklabels(xticks, rotation=45, ha='right', fontsize=18)
ax.set_xticklabels(xticks, rotation=0, ha="center", fontsize=16)
ax.set(ylabel="score")

baseline_score = weights.mean()
plt.axhline(baseline_score, linestyle="--", color="k", label="SE weighted mode")

plt.ylim(0, 1)
plt.legend()
plt.savefig("figures/cluster_entropy_weighted.tiff", bbox_inches="tight", dpi=600)
plt.show()
