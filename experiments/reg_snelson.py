import os
import sys
import itertools

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

import gpytorch
import pdb

# from plot_utils import tableau20

sys.path.append("../")
from gplml.models import (
    SGPR,
    CGLB,
    GPR,
    LowerBoundCG,
    LowerBoundSGPR,
    LowerBoundExact,
    logdet_estimator_awb,
    logdet_estimator_sgpr,
    logdet_estimator_exact,
    logdet_estimator_thang,
)

# note that this helper function does two different things:
# (i) plots the observed data;
# (ii) plots the predictions from the learned GP after conditioning on data;
# (iii) plots inducing inputs if given


def plot(
    plot_observed_data=False,
    plot_predictions=False,
    model=None,
    n_test=500,
    plot_inducing=False,
):

    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        plt.plot(X.numpy(), y.numpy(), "kx")

    if plot_predictions:
        Xtest = torch.linspace(-3, 10, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            # mean, var = model(Xtest, full_cov=False, full_output_cov=False)
            dist = model(Xtest)
            mean, cov = dist.mean.detach(), dist.covariance_matrix.detach()

        # sd = var.sqrt()  # standard deviation at each input point x
        sd = cov.diag().sqrt()
        plt.plot(Xtest.numpy(), mean.numpy(), "r", lw=2)  # plot the mean
        plt.fill_between(
            Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            color="C0",
            alpha=0.3,
        )
    if plot_inducing:
        Xu = model.covar_module.inducing_points
        # compute predictive mean and variance
        with torch.no_grad():
            # mean, var = model(Xu, full_cov=False, full_output_cov=False)
            dist = model(Xu)
            mean, cov = dist.mean.detach(), dist.covariance_matrix.detach()

        # sd = var.sqrt()  # standard deviation at each input point x
        sd = cov.diag().sqrt()
        plt.plot(Xu.data.numpy(), mean.numpy(), "s", color="r")  # plot the mean

    plt.xlim(-3, 10)
    plt.ylim(-4, 4)


X = np.loadtxt("data/train_inputs.txt")
# X = X.reshape(X.shape[0], 1)
X = torch.from_numpy(X).float()
y = np.loadtxt("data/train_outputs.txt")
# y = y.reshape(y.shape[0], 1)
y = torch.from_numpy(y).float()

data = (X, y)
N = X.shape[0]

seeds = np.arange(10)
Ms = [2]
num_steps = 5000
terms_avg = []
terms_exact = []
for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # initialize the inducing inputs
    terms_Ms = []
    for M in Ms:
        idx = torch.randperm(N)[:M]
        quad_term_estimators = ["sgpr", "cg", "exact"]
        logdet_term_estimators = ["sgpr", "awb", "thang", "exact"]
        methods = itertools.product(quad_term_estimators, logdet_term_estimators)
        terms_all = []
        for (quad_term, logdet_term) in methods:
            print(seed, M, quad_term, logdet_term)
            # Initialize the kernel and model.
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = 0.1

            # Base kernel is RBFKernel with scale and lengthscale params.
            rbf_kernel = gpytorch.kernels.RBFKernel()
            rbf_kernel.lengthscale = 2.0

            base_kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)
            base_kernel.outputscale = 1.0

            # Use InducingPointKernel wrapper.
            kernel = gpytorch.kernels.InducingPointKernel(
                base_kernel, inducing_points=X[idx], likelihood=likelihood
            )

            # pyro.clear_param_store()
            # kernel = gp.kernels.RBF(
            #     input_dim=1,
            #     variance=torch.tensor(1.0),
            #     lengthscale=torch.tensor(2.0),
            # )

            if logdet_term == "sgpr":
                logdet_estimator = logdet_estimator_sgpr
            elif logdet_term == "awb":
                logdet_estimator = logdet_estimator_awb
            elif logdet_term == "thang":
                logdet_estimator = logdet_estimator_thang
            elif logdet_term == "exact":
                logdet_estimator = logdet_estimator_exact

            if quad_term == "sgpr":
                model = SGPR(data, likelihood, kernel)
                lower_bound = LowerBoundSGPR(model, logdet_estimator)
            elif quad_term == "cg":
                model = CGLB(data, likelihood, kernel)
                lower_bound = LowerBoundCG(model, logdet_estimator)
            elif quad_term == "exact":
                model = SGPR(data, likelihood, kernel)
                lower_bound = LowerBoundExact(model, logdet_estimator)

            # sgpr = SparseGPRegression(
            #     X,
            #     y,
            #     kernel,
            #     Xu=X[idx],
            #     jitter=1.0e-4,
            #     approx=method,
            #     noise=torch.tensor(0.1),
            # )
            # Check parameters.
            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            terms = {
                "lower_bound": [],
                "quad_term": [],
                "logdet_term": [],
            }
            pbar = tqdm(range(num_steps))
            for i in pbar:
                optimizer.zero_grad()
                lb, quad, logdet = lower_bound.forward(data)
                loss = -lb
                loss.backward()
                optimizer.step()

                terms["lower_bound"].append(lb.item())
                terms["quad_term"].append(quad.item())
                terms["logdet_term"].append(logdet.item())

                pbar.set_postfix(
                    {
                        "lower_bound": lb.item(),
                        "quad_term": quad.item(),
                        "logdet_term": logdet.item(),
                    }
                )

            terms_all.append(terms)

            model.eval()
            likelihood.eval()

            # plot the predictions from the sparse GP
            plot(
                model=model,
                plot_observed_data=True,
                plot_predictions=True,
                plot_inducing=True,
            )
            plt.savefig(
                "./tmp/reg1d_snelson_quad_term_%s_logdet_term_%s_M_%d_seed_%d.pdf"
                % (quad_term, logdet_term, M, seed),
                bbox_inches="tight",
                pad_inches=0,
            )
        terms_Ms.append(terms_all)
    terms_avg.append(terms_Ms)

    plt.close("all")

terms_avg = {k: np.array(v) for k, v in terms_avg.items()}
terms_mean = {k: np.mean(v, 0) for k, v in terms_avg.items()}
terms_std = {k: np.std(v, 0) / np.sqrt(len(seeds)) for k, v in terms_avg.items()}
for j, M in enumerate(Ms):
    plt.figure()
    for i in range(len(methods)):
        plt.plot(
            np.arange(num_steps) + 1,
            terms_mean["lower_bound"][j, i, :],
            label=methods[i] + ", M=%d" % M,
            # color=tableau20[i * 2],
        )
        for k in range(len(seeds)):
            plt.plot(
                np.arange(num_steps) + 1,
                terms_avg["lower_bound"][k, j, i, :],
                alpha=0.1,
                # color=tableau20[i * 2],
            )
    # if j == len(Ms) - 1:
    #     plt.plot(
    #         np.arange(num_steps) + 1,
    #         loss_exact_mean,
    #         label="exact",
    #         color=tableau20[len(Ms) * 2],
    #     )
    #     for k in range(len(seeds)):
    #         plt.plot(
    #             np.arange(num_steps) + 1,
    #             loss_exact[k, :],
    #             color=tableau20[len(Ms) * 2],
    #             alpha=0.1,
    #         )
    plt.xlabel("iteration")
    plt.ylabel("objective")
    # plt.yscale("symlog")
    plt.xscale("log")
    plt.legend()
    plt.savefig(
        "/tmp/reg1d_snelson_loss_M_%d.pdf" % M,
        bbox_inches="tight",
        pad_inches=0,
    )
