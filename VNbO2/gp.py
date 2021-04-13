import gpflow
import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter


def fit_sparse_gp(X, lab, n_classes=2, optimize_lengthscales=True):
    """ use SVGP instead of VGP for better training scalability """

    # just use the the full combi grid for inducing points
    inducing_points = X[:192].copy()

    data = (X, lab)
    kernel = gpflow.kernels.RBF(
        lengthscales=[0.02, 10], variance=1.0
    ) + gpflow.kernels.White(variance=1.0)

    m = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.MultiClass(n_classes),
        inducing_variable=inducing_points,
        num_latent_gps=n_classes,
    )

    # choose a reasonable prior on the GP lengthscale over composition and temperature...
    m.kernel.kernels[0].lengthscales.prior = tfp.distributions.LogNormal(
        [np.log(0.02), np.log(10)], [1.0, 1.0]
    )

    # m.kern.white.variance.trainable = False
    if not optimize_lengthscales:
        m.kernel.kernels[0].lengthscales.trainable = False
        m.kernel.kernels[0].variance.trainable = False

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss_closure(data), m.trainable_variables, options={"maxiter": 1000}
    )

    return m


def variational_log_likelihoods(X, Y, model):
    """ compute variational expectations for a label set under the consensus model """
    # get predicted values to evaluate errors...
    Ymu, Yvar = model.predict_y(X)
    Ypred = np.argmax(Ymu, axis=1)
    error_indicator = 2 * (Y == Ypred) - 1

    # get individual log likelihoods
    Fmu, Fvar = model.predict_f(X)
    loglik = model.likelihood.variational_expectations(Fmu, Fvar, Y[:, None])

    return loglik, error_indicator


def snapshot(df, m, kelvin=False):
    cmap = plt.cm.get_cmap("YlGnBu_r", 3)

    max_C = 0.25
    C_spacing, T_spacing = 0.001, 0.1
    grid_C, grid_T = np.meshgrid(
        np.arange(0.0, max_C + C_spacing, C_spacing),
        np.arange(
            df.temp.min() - 50 * T_spacing, df.temp.max() + 50 * T_spacing, T_spacing
        ),
    )
    h, w = grid_C.shape
    gridpoints = np.c_[grid_C.ravel(), grid_T.ravel()]

    mu_y, var_y = m.predict_y(gridpoints)
    mu_f, var_f = m.predict_f(gridpoints)
    mu_y, var_y = mu_y.numpy().reshape(h, w, -1), var_y.numpy().reshape(h, w, -1)
    mu_f, var_f = mu_f.numpy().reshape(h, w, -1), var_f.numpy().reshape(h, w, -1)

    # color map the predictions... set alpha channel with variance...
    C = np.argmax(mu_y > 0.5, axis=-1).astype(int)
    c = cmap(C)
    a = var_y.mean(axis=-1)
    a = Normalize(a.min(), a.max(), clip=True)(a)
    c[..., -1] = 1 - a

    fig, ax1 = plt.subplots(ncols=1, figsize=(6, 4))

    if kelvin:
        grid_T += 273

    extent = (np.min(grid_C), np.max(grid_C), np.min(grid_T), np.max(grid_T))

    # ax1.scatter(df['Nb'], df['temp'])
    ax1.imshow(c, origin="lower", extent=extent)
    ax1.contour(
        C,
        levels=[0.5, 1.5],
        linestyles=["--"],
        colors="k",
        origin="lower",
        extent=extent,
    )
    ax1.set_xlabel("Nb")
    ax1.set_ylabel("Temperature (C)")
    if kelvin:
        ax1.set_ylabel("Temperature (K)")
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    plt.xlim(0, 0.25)
    plt.axis("auto")
    plt.tight_layout()
    return c, extent


def snapshot_gray(df, m):
    cmap = plt.cm.get_cmap("Blues")

    max_C = 0.25
    C_spacing, T_spacing = 0.001, 0.1
    grid_C, grid_T = np.meshgrid(
        np.arange(0.0, max_C + C_spacing, C_spacing),
        np.arange(
            df.temp.min() - 50 * T_spacing, df.temp.max() + 50 * T_spacing, T_spacing
        ),
    )
    h, w = grid_C.shape
    extent = (np.min(grid_C), np.max(grid_C), np.min(grid_T), np.max(grid_T))
    gridpoints = np.c_[grid_C.ravel(), grid_T.ravel()]

    mu_y, var_y = m.predict_y(gridpoints)
    mu_f, var_f = m.predict_f(gridpoints)
    mu_y, var_y = mu_y.numpy().reshape(h, w, -1), var_y.numpy().reshape(h, w, -1)
    mu_f, var_f = mu_f.numpy().reshape(h, w, -1), var_f.numpy().reshape(h, w, -1)

    # color map the predictions... set alpha channel with variance...
    c = var_y.mean(axis=-1)
    c = Normalize(c.min(), c.max(), clip=True)(c)
    C = np.argmax(mu_y > 0.5, axis=-1).astype(int)
    # c = cmap(C)

    # a = Normalize(a.min(), a.max(), clip=True)(a)
    # c[..., -1] = 1 - a

    fig, ax1 = plt.subplots(ncols=1, figsize=(6, 4))
    # ax1.scatter(df['Nb'], df['temp'])
    ax1.imshow(c, origin="lower", extent=extent, cmap="Greys")
    ax1.contour(
        C,
        levels=[0.5, 1.5],
        linestyles=["--"],
        colors="w",
        origin="lower",
        extent=extent,
    )
    ax1.set_xlabel("Nb")
    ax1.set_ylabel("Temperature (C)")
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    plt.xlim(0, 0.25)
    plt.axis("auto")
    plt.tight_layout()
    return c, extent
