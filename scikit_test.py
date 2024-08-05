"""
File: scikit_test.py
Author: Ming Creekmore
Purpose: To test the various surrogate models, acquisition functions, and initial point
         generators on the accuracy of select data and benchmark tests using scikit-optimize
"""
from timeit import default_timer
from skopt import Optimizer, sampler
import random
import matplotlib.pyplot as plt

from benchmarks import *
from scikit_plot import *

MATERIAL_TEMP = [("CU", 25.0), ("TP", 50.0), ("MN", 70.0), ("SN", 90.0), ("PC", 40.0), ("DR", 40.0)]
MATERIAL_DICT = {"CU": 25.0, "TP": 50.0, "MN": 70.0, "SN": 90.0, "PC": 40.0, "DR": 40.0}
func = dummy_measure(MATERIAL_TEMP)

def material_constraint(params):
    """
    Dummy boiling point constraint on materials
    """
    bp = MATERIAL_TEMP[params[2]][1]
    if bp < params[1] < bp + 20:
        return True
    return False

def plot_distribution(data, name):
    """
    Plotting a bar chart of the data, assuming the data is a list of categorical values
    """
    data_dict = dict()
    for d in data:
        if d in data_dict:
            data_dict[d] = data_dict[d] + 1
        else:
            data_dict[d] = 1
    plt.bar(list(data_dict.keys()), list(data_dict.values()))
    plt.title(name)
    plt.savefig(name)
    plt.show()

def test(model = "GP", model_name = "GP", base_dir = "./"):
    material_distribution = []
    directory = base_dir + model_name
    print("Model used: " + model_name)
    initial_point_gen = sampler.Sobol()
    opt = Optimizer([(5.0,50.0),(25.0,100.0), (0, 5)], base_estimator=model, acq_func='EI',
                    # n_initial_points=10, initial_point_generator="sobol",
                    space_constraint=material_constraint
    )
    
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
                material_distribution.append(p[2])
                n+=1
                if n == initial_points:
                    break
    # plot_distribution(material_distribution, model_name + " Initial Point Loop")
    # Perform experiment and save values in ys
    ys = []
    for p in points:
        ys.append(func(p))
    # tell the optimizer 
    opt.tell(points, ys)

    # main bayesian loop
    max_iter = 20
    for i in range(max_iter):
        
        # Get conditions to sample
        next_x = opt.ask()
        material_distribution.append(next_x[2])
        # Run the experiment at the given conditions
        if material_constraint(next_x):
            print("Trial", i, ": PASS")
        else:
            print("Trial", i, ": FAIL")
        print("NEXT:", next_x)
        # Process the raw data
        fval = func(next_x)

        # Update the optimizer with objective function value
        result = opt.tell(next_x, fval)
    duration = default_timer() - start
    # print(material_distribution)
    # plot_distribution(material_distribution, model_name + " Complete Distribution")

    print("Lowest minimum of " + str(result.fun) + " at " + str(result.x))
    print_results(result, directory)
    print("Time elapsed for loop:", duration)
    input("\nPress Enter to quit")

if __name__ == "__main__":
    test(model="GP", model_name="GP")
    # plotting 4d objective
    # plt4d(func, n=500)