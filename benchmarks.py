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

def dummy_measure(m_lst):
    def f(params):
        motor, heater, material = params
        bp = m_lst[material][1]
        if material <= 3:
            return math.sin(motor) + (heater - bp - 10)/10
        else:
            return math.sin(motor) - abs((heater - bp - 10)/10)
    return f

if __name__ == "__main__":
    print(rastrigan([0,1]))