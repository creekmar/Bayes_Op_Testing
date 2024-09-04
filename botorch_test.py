"""
File: botorch_test.py
Author: Ming Creekmore
Purpose: Botorch code to optimize photo td based on the parameters
         Motor Speed, Temp, Concentration, Print Gap, Vol.
         The solvent will determine the temperature bounds, rn CF is the bounds
"""

# Other imports
import time
import warnings
from scikit_plot import plot_optimization_trace
import pandas as pd
from constants import calc_temp_bounds

# Botorch imports
from botorch.models import MixedSingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf_mixed
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import (
     qNoisyExpectedImprovement,
)
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms import Normalize, Standardize
import torch

# discrete/categorical input space
SOLV_TEMP = [("CF", 61.2), ("Tol", 110.6), ("CB", 132), ("TCB", 214.4), ("DCB", 180.1)]
SOLV_NAMES = ["CF", "TOL", "CB", "TCB", "DCB"]
BP_SOLV = {61.2: "CF", 110.6: "TOL", 132: "CB", 214.4: "TCB", 180.1: "DCB"}
BP = [61.2, 110.6, 132, 214.4, 180.1]
CONCEN = [10, 15, 20]
PRINT_GAP = [25, 50, 75, 100]
PREC_VOL = [6, 9, 12]

# ML constants
NOISE_SE = 0.5
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Speed, Temp, Concentration, print gap, vol for chloroform
CF_BOUNDS = calc_temp_bounds("CF", (0, .9), (20, 140))
SPACE_BOUNDS = torch.tensor([(0.01, CF_BOUNDS[0], 10, 25, 6), (25.0, CF_BOUNDS[1], 20, 100, 12)], device=DEVICE)


###############################
# Specific for this problem
###############################
def make_feature_list():
    """@return a list of combinations of all choices in categorical/discrete parameters
       For the fixed_feature_list in botorch"""
    f_list = []
    for c in CONCEN:
        for g in PRINT_GAP:
            for v in PREC_VOL:
                f_list.append({2: c, 3: g, 4: v})

    return f_list


def dummy_measure(params):
    """@return dummy measure for the parameters motor speed, heater,
       concentration, printing gap, and precursor volume.
       An upside down parabola to minimize"""
    motor, heater, conc, gap, vol = params
    return (gap*motor +conc*pow(heater, 2) + vol)


def take_dummy_measures(data):
    """
    @return a tensor of the objective dummy measurement of a list of data
    """
    result = torch.zeros(len(data))
    for i in range(len(data)):
        result[i] = dummy_measure(data[i])
    return result

def transform(values):
    """
    @return a function that will get the given
        index in a list of values
    @param values: the list of values 
    """
    def f(num):
        """Returns the num index in a list of values"""
        return values[int(num)]
    return f


def generate_init_data(n=10, maximize = True):
    """
    Generates the initial random data for the problem space
    @param n: the number of data points to generate
    @return (train_X, train_Y, best_observed_value), where 
        train_X is the data points for the parameters, 
        train_Y is the measured objective points for the train_X data,
        best_observed_value is the highest train_Y 
    """
    b = torch.tensor([(0.01, CF_BOUNDS[0], 0, 0, 0), (25.0, CF_BOUNDS[1], 2, 3, 2)], device=DEVICE)
    raw_samples = draw_sobol_samples(bounds = b, n=1, q=n)
    train_X = raw_samples.flatten(0,1)

    # have to round and apply
    rounded = train_X[:,2:].round()
    rounded[:,0].apply_(transform(CONCEN))
    rounded[:,1].apply_(transform(PRINT_GAP))
    rounded[:,2].apply_(transform(PREC_VOL))
    train_X[:, 2:] = rounded

    # TODO replace with actual measure from experiment
    exact_obj = take_dummy_measures(train_X).unsqueeze(-1)  # add output dimension
    train_Y = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    if maximize:
        best_observed_value = train_Y.max().item()
    else:
        best_observed_value = train_Y.min().item()
    return train_X, train_Y, best_observed_value


###################
# ML FUNCS
###################
    
def initialize_model(train_x, train_obj, state_dict=None):
    """
    Create a GP Model and Marginal Log likelihood based on
    the data (motor speed, temperature, concentration, printing gap,
    precursor volume)
    @return (marginal_log_likelihood, GP_model)
    """

    # combine into a multi-output GP model
    model = MixedSingleTaskGP(train_x, train_obj, [2,3,4], outcome_transform=Standardize(m=1), input_transform=Normalize(d=5))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def optimize_acqf_and_get_observation(acq_func, batch_size=1, num_restarts=10, raw_samples=64):
    """
    Optimizes the cquisition function, and returns a new candidate and a noisy observation.
    @param acq_func: An instance of the acquisition function
    @param batch_size: the number of samples per batch
    @param num_restarts
    @param raw_samples
    """

    feat_list = make_feature_list()
    candidates, _ = optimize_acqf_mixed(
        acq_function=acq_func,
        bounds=SPACE_BOUNDS,
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
    
    # TODO Replace with actual measurement
    exact_obj = take_dummy_measures(new_x).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    return new_x, new_obj

def run_optimization(n_start_runs=10, n_loop_runs=50, mc_samples=256, maximize = True, verbose=True,
                     state_dict_file='model_state_dict.pth', 
                     data_file='lab_automation_data.csv'):
    """
    Full Bayesian Optimization loop using qNoisyExpectedImprovement for acquisition
    function.
    Optimizing a dummy measure of photo td, with parameters 
    ["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Objective"]
    Saves a graph of the optimization trace as png, and the model_state_dict
    """
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    best_observed_nei, iteration_list  = [], []
    time_list = torch.zeros(n_loop_runs+n_start_runs, 1)
    # call helper functions to generate initial training data and initialize model
    (
        train_x_nei,
        train_obj_nei,
        best_observed_value_nei,
    ) = generate_init_data(n_start_runs, maximize)
    mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei)

    best_observed_nei.append(best_observed_value_nei)

    # run N_RUNS runs of BayesOpt after the initial random data
    for iteration in range(1, n_loop_runs + 1):

        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_mll(mll_nei)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

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
        if maximize:
            best_value_nei = train_obj_nei.max().item()
            best_x_nei = train_x_nei[torch.argmax(train_obj_nei).item()]
        else:
            best_value_nei = train_obj_nei.min().item()
            best_x_nei = train_x_nei[torch.argmin(train_obj_nei).item()]
        best_observed_nei.append(best_value_nei)


        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            model_nei.state_dict(),
        )

        t1 = time.monotonic()
        time_list[iteration-1+n_start_runs] = t1-t0

        if verbose:
            print(
                f"\nRUN {iteration:>2}: \n\t"
                f"qNEI Best Value: {best_value_nei:>4.2f} at {best_x_nei.tolist()}\n\t"
                f"Time = {t1-t0:>4.2f}.\n",
                end="",
            )
        else:
            print(".", end="")

    # save data and plot optimization trace
    # TODO figure out how to save STATE_DICT
    torch.save(model_nei.state_dict(), state_dict_file)
    for i in range(len(best_observed_nei)):
        iteration_list.append(i)
    plot_optimization_trace(iteration_list, best_observed_nei)

    # save data as csv
    array = torch.cat((train_x_nei, train_obj_nei, time_list), 1).numpy()
    col_names = ["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Objective", "Time"]
    df = pd.DataFrame(array, columns=col_names)
    df.to_csv(data_file)
    print("Average time:", time_list.sum().item()/(n_loop_runs+n_start_runs))
    


if __name__ == "__main__":
    for solvent in SOLV_NAMES:
        temp_bounds = calc_temp_bounds("CF", (0, .9), (20, 140))
        space = torch.tensor([(0.01, temp_bounds[0], 10, 25, 6), (25.0, temp_bounds[1], 20, 100, 12)], device=DEVICE)
    run_optimization(maximize=False)
    


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