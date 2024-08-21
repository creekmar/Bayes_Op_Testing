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


import math
import os
from typing import Optional
import time
import warnings

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
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms import Normalize, Standardize

import torch

import numpy as np
from matplotlib import pyplot as plt
from scikit_plot import plot_optimization_trace

# TODO change to actual input space
SOLV_TEMP = [("CF", 61.2), ("Tol", 110.6), ("CB", 132), ("TCB", 214.4), ("DCB", 180.1)]
DUMMY_MATERIAL = [("CU", 25.0), ("TP", 50.0), ("MN", 70.0), ("SN", 90.0), ("PC", 40.0), ("DR", 40.0)]
BP_SOLV = {61.2: "CF", 110.6: "TOL", 132: "CB", 214.4: "TCB", 180.1: "DCB"}
BP = [61.2, 110.6, 132, 214.4, 180.1]
CONCEN = [10, 15, 20]
PRINT_GAP = [25, 50, 75, 100]
PREC_VOL = [6, 9, 12]
# FUNC = dummy_measure(DUMMY_MATERIAL)
NOISE_SE = 0.5
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# this needs to be in the shape of lows, then high
DUMMY_BOUNDS = torch.tensor([(5.0, 25.0, 0), (50.0, 100.0, 5)], device=DEVICE)
REAL_BOUNDS = torch.tensor([(0.0, 20.0), (25.0, 140.0)], device=DEVICE)

# Speed, Temp, Concentration, print gap, vol
CF_BOUNDS = torch.tensor([(0.01, 50.0, 10, 25, 6), (25.0, 60.0, 20, 100, 12)], device=DEVICE)


###############################
# Specific for this problem
###############################
def make_feature_list():
    # f_list = [{i: v} for v in lst]
    f_list = []
    for c in CONCEN:
        for g in PRINT_GAP:
            for v in PREC_VOL:
                f_list.append({2: c, 3: g, 4: v})

    return f_list

# print(make_feature_list())



def dummy_measure(params):
    motor, heater, conc, gap, vol = params
    return -(gap*motor +conc*pow(heater, 2) + vol)


def measure(data):
    result = torch.zeros(len(data))
    for i in range(len(data)):
        result[i] = dummy_measure(data[i])
    return result

def transform(values):
    def f(num):
        return values[int(num)]
    return f


def generate_init_data(n=10):
    b = torch.tensor([(0.01, 50.0, 0, 0, 0), (25.0, 60.0, 2, 3, 2)], device=DEVICE)
    raw_samples = draw_sobol_samples(bounds = b, n=1, q=n)
    train_X = raw_samples.flatten(0,1)

    # have to round and apply
    rounded = train_X[:,2:].round()
    rounded[:,0].apply_(transform(CONCEN))
    rounded[:,1].apply_(transform(PRINT_GAP))
    rounded[:,2].apply_(transform(PREC_VOL))
    train_X[:, 2:] = rounded
    exact_obj = measure(train_X).unsqueeze(-1)  # add output dimension
    train_Y = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    best_observed_value = train_Y.max().item()
    return train_X, train_Y, best_observed_value


###################
# ML FUNCS
###################
    
def initialize_model(train_x, train_obj, state_dict=None):
    # combine into a multi-output GP model
    model = MixedSingleTaskGP(train_x, train_obj, [2,3,4], outcome_transform=Standardize(m=1), input_transform=Normalize(d=5))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def optimize_acqf_and_get_observation(acq_func, batch_size=1, num_restarts=10, raw_samples=64):
    # Optimizes the acquisition function, and returns a new candidate and a noisy observation.
    # optimize
    feat_list = make_feature_list()
    candidates, _ = optimize_acqf_mixed(
        acq_function=acq_func,
        bounds=CF_BOUNDS,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        fixed_features_list=feat_list,
        # TODO later add in solvent
        # inequality_constraints=[(1, 1/1.8, BP)],
        #batch_initial_conditions=gen_batch_conditions(),
        # options={"batch_limit": 1, "maxiter": 200}
        )

   
    # observe new values
    new_x = candidates.detach()
    print("\tNew Candidate:", new_x)
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
    N_RUNS = 50
    MC_SAMPLES = 256

    verbose = True

    best_observed_nei, iteration_list  = [], []
    # call helper functions to generate initial training data and initialize model
    (
        train_x_nei,
        train_obj_nei,
        best_observed_value_nei,
    ) = generate_init_data(n=10)
    mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei)

    best_observed_nei.append(best_observed_value_nei)

    # run N_RUNS runs of BayesOpt after the initial random data
    for iteration in range(1, N_RUNS + 1):

        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_mll(mll_nei)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # for best_f, we use the best observed noisy values as an approximation
        qNEI = qNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=train_x_nei,
            sampler=qmc_sampler,
        )

        # optimize and get new observation
        new_x_nei, new_obj_nei = optimize_acqf_and_get_observation(qNEI)

        # update training points
        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])

        # update progress
        best_value_nei = train_obj_nei.max().item()
        best_observed_nei.append(best_value_nei)

        best_x_nei = train_x_nei[torch.argmax(train_obj_nei).item()]

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            model_nei.state_dict(),
        )
        # model_nei.state_dict()

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nRUN {iteration:>2}: \n\t"
                f"qNEI Best Value: {best_value_nei:>4.2f} at {best_x_nei.tolist()}\n\t"
                f"Time = {t1-t0:>4.2f}.\n",
                end="",
            )
        else:
            print(".", end="")

        # NOTE: there were 13 in the list
    # TODO figure out how to save STATE_DICT
    torch.save(model_nei.state_dict(), 'model_state_dict.pth')
    for i in range(len(best_observed_nei)):
        iteration_list.append(i)
    plot_optimization_trace(iteration_list, best_observed_nei)


if __name__ == "__main__":
    run_optimization()
    


# """
####################################
# GRAVEYARD
####################################

# def gen_batch_conditions(num=10):
#     mat = 0
#     count = 0
#     total_samples = torch.empty(1, num,3)
#     while count < num:
#         # temp constraints
#         bp = DUMMY_MATERIAL[mat][1]
#         temp_high = min(100, bp*1.8)
#         b = torch.tensor([(5.0, 25.0, mat), (50.0, temp_high, mat)], device=DEVICE)

#         # for some reason draw_sobol_samples adds an extra dimension so need to flatten
#         raw_samples = draw_sobol_samples(bounds = b, n=1, q=1)
#         samples = raw_samples.flatten(0,1)
#         total_samples[0,count] = samples
#         count += 1

#         # have material change every time
#         mat = (mat + 1)%6
#     return total_samples

# def generate_dummy_init_data(n=10):
#     train_X = gen_batch_conditions(n).flatten(0,1)
#     exact_obj = measure(train_X).unsqueeze(-1)  # add output dimension
#     train_Y = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
#     best_observed_value = train_Y.max().item()
#     return train_X, train_Y, best_observed_value