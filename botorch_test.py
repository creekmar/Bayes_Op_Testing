#!/usr/bin/env python3
# coding: utf-8

# ## Closed-loop batch, constrained BO in BoTorch with qEI and qNEI
# 
# In this tutorial, we illustrate how to implement a simple Bayesian Optimization (BO) closed loop in BoTorch.
# 
# In general, we recommend for a relatively simple setup (like this one) to use Ax, since this will simplify your setup (including the amount of code you need to write) considerably. See the [Using BoTorch with Ax](./custom_botorch_model_in_ax) tutorial.
# 
# However, you may want to do things that are not easily supported in Ax at this time (like running high-dimensional BO using a VAE+GP model that you jointly train on high-dimensional input data). If you find yourself in such a situation, you will need to write your own optimization loop, as we do in this tutorial.
# 
# 
# We use the batch Expected Improvement (qEI) and batch Noisy Expected Improvement (qNEI) acquisition functions to optimize a constrained version of the synthetic Hartmann6 test function. The standard problem is
# 
# $$f(x) = -\sum_{i=1}^4 \alpha_i \exp \left( -\sum_{j=1}^6 A_{ij} (x_j - P_{ij})^2  \right)$$
# 
# over $x \in [0,1]^6$ (parameter values can be found in `botorch/test_functions/hartmann6.py`).
# 
# In real BO applications, the design $x$ can influence multiple metrics in unknown ways, and the decision-maker often wants to optimize one metric without sacrificing another. To illustrate this, we add a synthetic constraint of the form $\|x\|_1 - 3 \le 0$. Both the objective and the constraint are observed with noise. 
# 
# Since botorch assumes a maximization problem, we will attempt to maximize $-f(x)$ to achieve $\max_{x} -f(x) = 3.32237$.

# In[1]:


import os
from typing import Optional
import time
import warnings
from benchmarks import dummy_measure

# Botorch imports
from botorch.models import MixedSingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf_mixed

from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms import Normalize, Standardize

import torch

import numpy as np
from matplotlib import pyplot as plt


MATERIAL_TEMP = [("CU", 25.0), ("TP", 50.0), ("MN", 70.0), ("SN", 90.0), ("PC", 40.0), ("DR", 40.0)]
FUNC = dummy_measure(MATERIAL_TEMP)
NOISE_SE = 0.5
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# this needs to be in the shape of lows, then highs
BOUNDS = torch.tensor([(5.0, 25.0, 0), (50.0, 100.0, 5)], device=DEVICE)

# train_x, train_obj, train_con, best_observed_value = generate_initial_data()
# print("Train_x:", train_x)
# print("Train_obj:", train_obj)
# print("Train_con:", train_con)
# print("Weighted:", weighted_obj(train_x).unsqueeze(-1))
# print("Best objserved value:", best_observed_value)

###############################
# Specific for this problem
###############################
def measure(data):
    result = torch.zeros(len(data))
    for i in range(len(data)):
        result[i] = FUNC(data[i])
    return result

def material_constraint(params):
    """
    Dummy boiling point constraint on materials
    """
    bp = MATERIAL_TEMP[int(params[2])][1]
    if params[1] < bp * 1.8:
        return True
    return False



# TODO Figure out wtf this is
def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    # get 0th column
    return Z[..., 0]


def constraint_callable(Z):
    # get 1st column
    return Z[..., 1]


# define a feasibility-weighted objective for optimization
constrained_obj = ConstrainedMCObjective(
    objective=obj_callable,
    constraints=[constraint_callable],
)


###################
# Generate data
###################

def generate_init_data(n=10):
    # TODO add constraints, make more general
    train_X = torch.cat(
            [torch.rand(n, 1) * 45+5, torch.rand(n, 1) * 75+25, 
            torch.randint(6, (n, 1))], dim=-1
            )
    
    exact_obj = measure(train_X).unsqueeze(-1)  # add output dimension
    train_Y = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    best_observed_value = train_Y.max().item()
    return train_X, train_Y, best_observed_value

def initialize_model(train_x, train_obj, state_dict=None):
    # combine into a multi-output GP model
    model = MixedSingleTaskGP(train_x, train_obj, [2], outcome_transform=Standardize(m=1), input_transform=Normalize(d=3))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def optimize_acqf_and_get_observation(acq_func, batch_size=3, num_restarts=10, raw_samples=64):
    # Optimizes the acquisition function, and returns a new candidate and a noisy observation.
    # optimize
    found = False
    while not found:
        candidates, _ = optimize_acqf_mixed(
            acq_function=acq_func,
            bounds=BOUNDS,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,  # used for intialization heuristic
            fixed_features_list=[{2: v} for v in range(6)],
        )
        # TODO figure out constraints
        # if material_constraint(candidates.detach()):
        #     found = True
        found = True
        
    # observe new values
    new_x = candidates.detach()
    exact_obj = measure(new_x).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    return new_x, new_obj



# ### Perform Bayesian Optimization loop with qNEI
# The Bayesian optimization "loop" for a batch size of $q$ simply iterates the following steps:
# 1. given a surrogate model, choose a batch of points $\{x_1, x_2, \ldots x_q\}$
# 2. observe $f(x)$ for each $x$ in the batch 
# 3. update the surrogate model. 
# 
# 
# Just for illustration purposes, we run three trials each of which do `N_BATCH=20` rounds of optimization. The acquisition function is approximated using `MC_SAMPLES=256` samples.
# 
# *Note*: Running this may take a little while.

# In[6]:

def run_optimization():
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    N_BATCH = 20
    MC_SAMPLES = 256

    verbose = True

    best_observed_ei, best_observed_nei  = [], []

    # call helper functions to generate initial training data and initialize model
    (
        train_x_ei,
        train_obj_ei,
        best_observed_value_ei,
    ) = generate_init_data(n=10)
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)

    train_x_nei, train_obj_nei= train_x_ei, train_obj_ei
    best_observed_value_nei = best_observed_value_ei
    mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei)

    best_observed_ei.append(best_observed_value_ei)
    best_observed_nei.append(best_observed_value_nei)

    # TODO get rid of batched runs
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):

        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_mll(mll_ei)
        fit_gpytorch_mll(mll_nei)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # for best_f, we use the best observed noisy values as an approximation
        qEI = qExpectedImprovement(
            model=model_ei,
            best_f=(train_obj_ei * (train_obj_ei <= 0).to(train_obj_ei)).max(),
            sampler=qmc_sampler,
        )

        qNEI = qNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=train_x_nei,
            sampler=qmc_sampler,
        )

        # optimize and get new observation
        new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(qEI)
        new_x_nei, new_obj_nei = optimize_acqf_and_get_observation(qNEI)

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])

        # update progress
        best_value_ei = train_obj_ei.max().item()
        best_value_nei = train_obj_nei.max().item()
        best_observed_ei.append(best_value_ei)
        best_observed_nei.append(best_value_nei)
        # print("HELP:", torch.argmax(train_obj_ei), torch.argmax(train_obj_ei).item())

        best_x_ei = train_x_ei[torch.argmax(train_obj_ei).item()]
        best_x_nei = train_x_nei[torch.argmax(train_obj_nei).item()]

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei,
            train_obj_ei,
            model_ei.state_dict(),
        )
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            model_nei.state_dict(),
        )

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: \n\tqEI Best Value {best_value_ei:>4.2f}"
                f" at {best_x_ei.tolist()}\n\t"
                f"qNEI Best Value: {best_value_nei:>4.2f} at {best_x_nei.tolist()}\n\t"
                f"Time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")


if __name__ == "__main__":
    run_optimization()


# """