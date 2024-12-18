"""
File: scikit_test.py
Author: Ming Creekmore
Purpose: Bayesian Optimization applied to the problem space 
         [Speed, Temp, Concentration, print gap, vol, solvent]
         Dummy evaluation until integration with robot
"""
from timeit import default_timer
import numpy as np
import pandas as pd
from skopt import Optimizer

from benchmarks import dummy_measure
from constants import get_all_press_solv_bounds, make_pressure_from_temp_f, to_Kelvin
from initial_point_generator import get_continuous_sobol_initial_points
from scikit_plot import print_results

# discrete/categorical input space
CONCEN = (1,5) # this will be mapped to the equation 2n-1
PREC_VOL = (6.0, 12.0)
SOLV = ["CF", "CB", "CB9:A1", "CB8:A2", "CB7:A3"]
SPEED = (0.01, 20.0, "log-uniform") # make sure speed is logarithmic
PRESSURE = (0.0, 0.5)
TEMP_C = (25.0, 140.0)
SOLV_PRESS_BOUNDS = get_all_press_solv_bounds(SOLV, TEMP_C, PRESSURE)

SPACE_LABELS = ["Motor Speed", "Pressure", "Precursor Volume", 
                "Concentration", "Solvent"]

# Speed, Pressure, vol, concentration, solvent
SPACE = [SPEED, PRESSURE, PREC_VOL, CONCEN, SOLV]


def material_constraint(params):
    """
    Pressure constraint based on materials vp vs temp curve
    If pressure suggested is greater than the min, then true
    @params list of space params used in the problem
    """
    if SOLV_PRESS_BOUNDS[params[4]][0] < params[1]:
        return True
    return False


def test(model = "GP", model_name = "GP", base_dir = "./"):
    directory = base_dir + model_name
    print("Model used: " + model_name)
    
    # Space on [pressure, motor speed, precursor volume, concentration, solvent]
    opt = Optimizer(SPACE, 
                    base_estimator="RF", # GBT or ET, can try GP 
                    acq_func='UCB', # or EI
                    space_constraint=material_constraint)
    
    start = default_timer()
    initial_points=14
    points = get_sobol_initial_points(initial_points, True)
    
    # Perform experiment and save values in ys
    ys = []
    for p in points:
        ys.append(dummy_measure(p))
    # tell the optimizer 
    opt.tell(points, ys)

    # main bayesian loop
    max_iter = 20
    ts = np.zeros(initial_points+max_iter)
    for i in range(max_iter):
        t0 = default_timer()
        # Get conditions to sample
        next_x = opt.ask()
        # Run the experiment at the given conditions
        print("NEXT:", next_x)
        # Process the raw data
        fval = dummy_measure(next_x)
        t1 = default_timer() - t0
        ts[initial_points+i] = t1

        # Update the optimizer with objective function value
        result = opt.tell(next_x, fval)
    duration = default_timer() - start

    print("Lowest minimum of " + str(result.fun) + " at " + str(result.x))
    print_results(result, directory)
    print("Time elapsed for loop:", duration)
    # create dataframe to put in csv
    df = pd.DataFrame(opt.Xi, columns=SPACE_LABELS)
    df["Objective"] = opt.yi
    df["Time"] = ts
    df.to_csv("Scikit_Data.csv")
    # print("Dataframe", df)

if __name__ == "__main__":
    test(model="GP", model_name="GP")