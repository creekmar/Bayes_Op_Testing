"""
File: scikit_test.py
Author: Ming Creekmore
Purpose: To test the various surrogate models, acquisition functions, and initial point
         generators on the accuracy of select data and benchmark tests
"""


from skopt import Optimizer

from ml_testing.benchmarks import rastrigan
from ml_testing.scikit_plot import plot_true_objective, print_results


def test(model = "GP", base_dir = "./"):
    directory = base_dir + model
    print("Model used: " + model)
    opt = Optimizer([(-2.0,2.0),(-2.0,2.0)], base_estimator=model, acq_func='EI', n_initial_points=10, 
                    initial_point_generator='lhs', random_state=None)
    max_iter = 30
    for i in range(max_iter):
        
        # Get conditions to sample
        next_x = opt.ask()
        
        # Run the experiment at the given conditions

        # Process the raw data
        fval = rastrigan(next_x)

        # Update the optimizer with objective function value
        result = opt.tell(next_x, fval)
    
    plot_true_objective([(-2,2),(-2,2)], rastrigan, directory=directory)
    print_results(result, directory)
    input("\nPress Enter to quit")