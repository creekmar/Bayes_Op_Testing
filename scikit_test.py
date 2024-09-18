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
from constants import get_all_solvent_bounds
from initial_point_generator import get_sobol_initial_points
from scikit_plot import print_results

# discrete/categorical input space
BP_SOLV = {61.2: "CF", 110.6: "TOL", 132: "CB", 214.4: "TCB", 180.1: "DCB"}
BP = [61.2, 110.6, 132, 214.4, 180.1]
CONCEN = [10, 15, 20]
PRINT_GAP = [25, 50, 75, 100]
PREC_VOL = [6, 9, 12]

# Speed, Temp, Concentration, print gap, vol, solvent
SPACE = [(0.01,25.0),(20.0,140.0), CONCEN, PRINT_GAP, PREC_VOL, BP]
SOLV_TEMP_BOUNDS = get_all_solvent_bounds()

def material_constraint(params):
    """
    Dummy boiling point constraint on materials
    """
    bounds = SOLV_TEMP_BOUNDS[BP_SOLV[params[5]]]
    if bounds[0] <= params[1] <= bounds[1]:
        return True
    return False


def test(model = "GP", model_name = "GP", base_dir = "./"):
    directory = base_dir + model_name
    print("Model used: " + model_name)
    opt = Optimizer(SPACE, base_estimator=model, acq_func='EI',
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
    col_names = ["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Solvent"]
    df = pd.DataFrame(opt.Xi, columns=col_names)
    df["Objective"] = opt.yi
    df["Time"] = ts
    df['Solvent'] = df['Solvent'].replace(BP_SOLV)
    df.to_csv("Scikit_Data.csv")
    # print("Dataframe", df)

if __name__ == "__main__":
    test(model="GP", model_name="GP")