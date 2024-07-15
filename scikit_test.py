"""
File: scikit_test.py
Author: Ming Creekmore
Purpose: To test the various surrogate models, acquisition functions, and initial point
         generators on the accuracy of select data and benchmark tests using scikit-optimize
"""
from timeit import default_timer
from skopt import Optimizer, sampler

from benchmarks import *
from scikit_plot import *

MATERIAL_TEMP = [("CU", 20.0), ("TP", 50.0), ("MN", 70.0), ("SN", 90.0), ("PC", 40.0), ("DR", 40.0)]
func = dummy_measure(MATERIAL_TEMP)

def material_constraint(params):
    bp = MATERIAL_TEMP[params[2]][1]
    if bp < params[1] < bp + 20:
        return True
    return False

def test(model = "GP", model_name = "GP", base_dir = "./"):
    directory = base_dir + model_name
    print("Model used: " + model_name)
    initial_point_gen = sampler.Sobol()
    opt = Optimizer([(5.0,50.0),(25.0,100.0), (0, 5)], base_estimator=model, acq_func='EI',
                    # n_initial_points=10, initial_point_generator="sobol",
                    random_state=None, space_constraint=material_constraint)
    
    start = default_timer()
    initial_points = 10
    n = 0
    points = []
    
    # initial point loop
    while n < initial_points:
        x = initial_point_gen.generate([(5.0,50.0),(25.0,100.0), (0, 5)], 16)
        for p in x:
            if material_constraint(p):
                points.append(p)
                n+=1
                if n == initial_points:
                    break
    ys = []
    for p in points:
        ys.append(func(p))
    opt.tell(points, ys)

    # main loop
    max_iter = 20
    for i in range(max_iter):
        
        # Get conditions to sample
        next_x = opt.ask()
        
        # Run the experiment at the given conditions

        # Process the raw data
        fval = func(next_x)

        # Update the optimizer with objective function value
        result = opt.tell(next_x, fval)
    duration = default_timer() - start
    # plot_true_objective([(-10,10),(-10,10)], func, directory=directory)
    print("Time:", duration)
    print("Lowest minimum of " + str(result.fun) + " at " + str(result.x))
    # print_results(result, directory)
    input("\nPress Enter to quit")

if __name__ == "__main__":
    test(model="ET", model_name="ET")
    # plot_true_objective([(-10,10),(-10,10)], func)
    # plt4d(func, 200)