"""
log10(P) = A - (B / (T + C))
P = Pressure in Bar
T = Temp in K

Kelvin = Celsius + 273.15

These are the solvents we are interested in

CF - chloroform - 61.2                  A=4.56992   B=1486.455  C=-8.612        50-60 degrees rn
Tol - toluene - 110.6                   A=4.08245   B=1346.382  C=-53.508
CB - chlorobenzene - 132                A=4.11083   B=1435.675  C=-55.124
TCB - trichlorobenzene - 214.4          A=4.64002   B=2110.983	C=-30.721
DCB - dichlorobenzene - 180.1           A=4.19518   B=1649.55   C=-59.836

Not above 80-85% of boiling point

printing speed : more closely parse <1mm/s region and a bit far away at >1mm/s region. 
maybe worth trying normalizing in logarithmic way.
"""

from math import log
from benchmarks import dummy_measure
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

CONSTANTS = {"CF": [4.56992, 1486.455, -8.612], "TOL": [4.08245, 1346.382, -53.508],
             "CB": [4.11083, 1435.675, -55.124], "TCB": [4.64002, 2110.983, -30.721],
             "DCB": [4.19518, 1649.55, -59.836]}
CONCEN = [10, 15, 20]
PRINT_GAP = [25, 50, 75, 100]
PREC_VOL = [6, 9, 12]
NOISE_SE = 0.5


def make_pressure_from_temp_func(a, b, c):
    def f(temp):
        t = a-(b/(temp+c))
        return pow(10, t)
    return f

def make_temp_from_pressure_func(a, b, c):
    def f(pressure):
        t = b/(a - log(pressure, 10)) - c
        return t
    return f

def to_Kelvin(c_temp):
    return c_temp + 273.15

def to_Celsius(k_temp):
    return k_temp - 273.15

def plot_temp_pressure(temps, pressures, constants, name):
    get_pressure = make_pressure_from_temp_func(*constants)
    get_temp = make_temp_from_pressure_func(*constants)
    all_temps = []
    all_pressures = []
    for item in temps:
        # print("(temp, pressure):", item, get_pressure(item))
        all_temps.append(item)
        all_pressures.append(get_pressure(item))
    for item in pressures:
        # print("(pressure, temp):", item, get_temp(item))
        all_pressures.append(item)
        all_temps.append(get_temp(item))
    for i in range(len(all_pressures)):
        print("(temp, pressure):", all_temps[i], all_pressures[i])
    # return all_temps, all_pressures
    plt.scatter(all_temps, all_pressures)
    plt.savefig(name)


if __name__ == "__main__":
    for solvent in CONSTANTS:
        f = make_temp_from_pressure_func(*CONSTANTS[solvent])
        print(solvent, "Boiling point:", to_Celsius(f(.7)))

    # constants = [4.20772, 1233.129, -40.953]
    # other_constants = [4.56992, 1486.455, -8.612]
    # temps = [425, 450, 475, 500, 525]
    # pressures = [10, 20, 30, 40, 50, 60]
    # plot_temp_pressure(temps, pressures, constants, "first")
    # plot_temp_pressure(temps, pressures, other_constants, "second")
    
    
    