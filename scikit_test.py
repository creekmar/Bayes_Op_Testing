"""
File: scikit_test.py
Author: Ming Creekmore
Purpose: To test the various surrogate models, acquisition functions, and initial point
         generators on the accuracy of select data and benchmark tests using scikit-optimize
"""


from skopt import Optimizer

from benchmarks import *
from scikit_plot import plot_true_objective, print_results

func = rastrigan

def test(model = "GP", model_name = "GP", base_dir = "./"):
    directory = base_dir + model_name
    print("Model used: " + model_name)
    opt = Optimizer([(-10.0,10.0),(-10.0,10.0)], base_estimator=model, acq_func='EI', n_initial_points=10, 
                    initial_point_generator='lhs', random_state=None)
    max_iter = 30
    for i in range(max_iter):
        
        # Get conditions to sample
        next_x = opt.ask()
        
        # Run the experiment at the given conditions

        # Process the raw data
        fval = func(next_x)

        # Update the optimizer with objective function value
        result = opt.tell(next_x, fval)
    
    plot_true_objective([(-10,10),(-10,10)], func, directory=directory)
    print_results(result, directory)
    input("\nPress Enter to quit")

if __name__ == "__main__":
    test()