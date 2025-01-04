"""
File: benchmarks.py
Author: Ming Creekmore
Purpose: Provides mathematical functions to act as benchmark tests for any machine
         learning algorithms and hyperparameter tuning testing
"""

import math

def rastrigan(lst):
    """
    Gives the output of the rastrigan function given the inputs. Can be any dimension.
    Note: the minimum is always at (0,...,0)
    @param: lst - list of inputs
    @return: the evaluation of the inputs
    """
    return 10 * len(lst) + sum([(x**2 - 10 * math.cos(2 * math.pi * x)) for x in lst])

def rosenbrock(lst):
    """
    Gives the output of the rosenbrock function given the inputs. Can be any dimension
    Note: the minimum is always at (1,...,1)
    @param: lst - list of inputs
    @return: the evaluation of the inputs
    """
    d = len(lst)
    sum = 0
    for i in range(d-1):
        sum += 100*(lst[i+1] - lst[i]**2)**2 + (lst[i]-1)**2
    return sum

def mix(lst):
    return rastrigan(lst) + rosenbrock(lst)

def dummy_measure(params):
    """@return dummy measure for the parameters motor speed, heater,
       precursor volume, concentration, solvent/boiling point.
       An parabola to minimize
       The minimum is where all parameters are at their min
       1.011536 is the minimum with the current workspace"""
    motor, heater, vol, conc, bp = params
    print(params)

    return (motor +conc*pow(heater, 2) + vol)

if __name__ == "__main__":
    print(dummy_measure([0.01, 0.016, 1, 6, 'CF']))