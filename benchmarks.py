"""
File: benchmarks.py
Author: Ming Creekmore
Purpose: Provides mathematical functions to act as benchmark tests for any machine
         learning algorithms and hyperparameter tuning testing
"""

import math


def rastrigan(lst):
    return 10 * len(lst) + sum([(x**2 - 10 * math.cos(2 * math.pi * x)) for x in lst])