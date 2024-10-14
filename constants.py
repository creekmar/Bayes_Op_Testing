"""
log10(P) = A - (B / (T + C))
P = Pressure in Bar
T = Temp in K

Kelvin = Celsius + 273.15

These are the solvents we are interested in

CF - chloroform - 61.2                  A=4.20772   B=1233.129  C=-40.953        good
Tol - toluene - 110.6                   A=4.08245   B=1346.382  C=-53.508        good
CB - chlorobenzene - 132                A=4.11083   B=1435.675  C=-55.124        good
MXY - m-xylene - 139                    A=4.13607   B=1463.218  C=-57.991        good????
ANI - anisole - 154                     A=4.17726   B=1489.756  C=-69.607	     good
MES - mesitylene - 164.7                A=4.19927   B=1569.622  C=-63.572
DEC - decane - 174.1                    A=4.07857   B=1501.268	C=-78.67         good????
DCB - dichlorobenzene - 180.1           A=4.19518   B=1649.55   C=-59.836        good
TCB - trichlorobenzene - 214.4          A=4.64002   B=2110.983	C=-30.721        good


National Institute of Standards and Technology. NIST. (2024, August 29). https://www.nist.gov/ 

Not above 80-85% of boiling point

printing speed : more closely parse <1mm/s region and a bit far away at >1mm/s region. 
maybe worth trying normalizing in logarithmic way.

Temperature Bounds where temperature min is 20 and max is based on .9 pressure or 140 (whatever is lower)
CF (25, 41.2)
TOL (25, 90.6)
CB (25, 112)
TCB (25, 140)
DCB (25, 140)
DEC (25, 140)
"""

from math import log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONSTANTS = {"CF": [4.20772, 1233.129, -40.953], "TOL": [4.08245, 1346.382, -53.508],
             "CB": [4.11083, 1435.675, -55.124], "mXY": [4.13607, 1463.218, -57.991],
             "MES": [4.19927, 1569.622, -63.572], "DEC": [4.07857, 1501.268, -78.67], 
             "ANI": [4.17726, 1489.756, -69.607], #"TCB": [4.64002, 2110.983, -30.721], 
             "DCB": [4.19518, 1649.55, -59.836] }
M_XYL_CONSTANTS = {"Pitzer and Scott":[(273, 333),(5.09199, 1996.545, -14.772)],
                 "Williamham": [(332.4, 413.19), (4.13607, 1463.218, -57.991)]}
CONCEN = [10, 15, 20]
PRINT_GAP = [25, 50, 75, 100]
PREC_VOL = [6, 9, 12]


def make_pressure_from_temp_func(a, b, c):
    """
    @return a function that will give the pressure
    in bars given a temperature in Kelvin based
    on the vapor pressure vs temperature function
    @param a: constant a
    @param b: constant b
    @param c: constant c
    """
    def f(temp):
        """ Returns the pressure (bars) given the temperature (Kelvin) """
        t = a-(b/(temp+c))
        return pow(10, t)
    return f

def make_temp_from_pressure_func(a, b, c):
    """
    @return a function that will give the temperature
    in Kelvin given a pressure in bars based
    on the vapor pressure vs temperature function
    @param a: constant a
    @param b: constant b
    @param c: constant c
    """
    def f(pressure):
        """ Returns the temperature (Kelvin) given the pressure (bars) """
        t = b/(a - log(pressure, 10)) - c
        return t
    return f

def to_Kelvin(c_temp):
    return c_temp + 273.15

def to_Celsius(k_temp):
    return k_temp - 273.15

def plot_temp_pressure(temps, pressures, constants, name):
    """
    Plots vapor pressure (bars) vs temperature (Kelvin) graph
    @param temps: the temperatures to include in the plot, their
        pressures will be calculated based on the formula and constants
    @param pressures: the pressures to include in the plot, their
        temperatures will be calculated based on the formula and constants
    @param constants: the constants that go to the vp vs temp formula
    @param name: the name of the solvent we are plotting
    """
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
    plt.title(name + " VP vs Temp")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (bars)")
    plt.scatter(all_temps, all_pressures)
    plt.savefig(name)

def make_temp_from_pressure_f(solvent):
    """
    @return a function that will give the temperature
    in Kelvin given a pressure in bars for a specified solvent
    @param: solvent: the name of the solvent in CONSTANTS
    """
    f = make_temp_from_pressure_func(*CONSTANTS[solvent])
    return f

def calc_temp_bounds(solvent, percent_window, bounds):
    """
    Gives the temperature based on low and high pressure bounds,
    where pressure is based on bar and temperature is converted to
    Celsius
    @param solvent: the name of the solvent in dictionary CONSTANTS
    @param percent_window: tuple of the percent of pressure bounds
        if low is 0, then the temperature bound will be used instead
    @param bounds: tuple of the temperature bounds in Celsius
    @return: tuple of low and high temp bounds in Celsius
    """
    f = make_temp_from_pressure_func(*CONSTANTS[solvent])

    # Calculate high bound
    high = to_Celsius(f(percent_window[1]))
    if high > bounds[1]:
        high = bounds[1]

    # Calculate low bound
    low = 20
    if percent_window[0] != 0:
        temp = to_Celsius(f(percent_window[0]))
        if percent_window[0] < low < high:
            low = temp
    return low, high

def calc_pressure_bounds(solvent, temp_window):
    """
    Calculates pressure bounds given temperature in 
    Celsius, where the max is pressure of 1 bar
    @param solvent: the solvent name from the CONSTANTS dictionary
    @param temp_window: tuple of the low and high bounds of the
        temperature in Celsius
    @return tuple of low and high pressure bounds in bar
    """
    get_pressure = make_pressure_from_temp_func(*CONSTANTS[solvent])
    get_temp = make_temp_from_pressure_func(*CONSTANTS[solvent])
    max = to_Celsius(get_temp(1)) - 20
    if temp_window[1] > max:
        high = get_pressure(to_Kelvin(max))
    else:
        high = get_pressure(to_Kelvin(temp_window[1]))
    low = get_pressure(to_Kelvin(temp_window[0]))
    return (low, high)

def get_all_temp_solv_bounds():
    """
    Get a dictionary of the solvnt temperature bounds relating 
    solvent name to solvent temperature bound in C
    """
    bound_dict = {}
    for solvent in CONSTANTS:
        b = calc_temp_bounds(solvent, (0, .9), (25, 140))
        bound_dict[solvent] = b
    return bound_dict

def get_all_press_solv_bounds(low_temp = 25, high_temp = 140):
    """
    Get a dictionary of the solvnt pressure bounds relating 
    solvent name to solvent pressure bound in bars
    """
    bound_dict = {}
    for solvent in CONSTANTS:
        bound_dict[solvent] = calc_pressure_bounds(solvent, (low_temp, high_temp))
    return bound_dict

if __name__ == "__main__":

    # for solvent in CONSTANTS:
        # this is what we're using for the problem
        # print(solvent, ":", calc_pressure_bounds(solvent, (25, 140)))

    # ###############################################
    # # Temp Pressures of different curves
    # ###############################################
    # temp_press_dict = {"Pitzer and Scott":[], "Williamham": []}
    # for name in temp_press_dict:
    #     constants = M_XYL_CONSTANTS[name][1]
    #     f = make_temp_from_pressure_func(*constants)
    #     for press in pressures:
    #         temp_press_dict[name].append(to_Celsius(f(press)))
    # temp_press_dict["Pressure"] = pressures
    # df = pd.DataFrame(temp_press_dict)
    # print(df)
        
    ######################
    # Pressure choices
    ######################
    temp_press_list = []
    pressures =  [0.1, 0.2, 0.3, 0.4, 0.5]
    for solvent in CONSTANTS:

        # this is for getting a list of temperatures related to a lits of pressures
        f = make_temp_from_pressure_func(*CONSTANTS[solvent])
        for press in pressures:
            temp_press_list.append([solvent, press, to_Celsius(f(press))])
    df = pd.DataFrame(temp_press_list, columns=["Solvent", "Pressure (Bars)", "Temperature (Celsius)"])
    df.to_csv("Temp_Pressure_Choices.csv")
    print(df)

    #####################
    # Plotting Curves
    #####################
    # constants = [4.20772, 1233.129, -40.953]
    # other_constants = [4.56992, 1486.455, -8.612]
    # temps = [425, 450, 475, 500, 525]
    # pressures = [10, 20, 30, 40, 50, 60]
    # plot_temp_pressure(temps, pressures, constants, "first")
    # plot_temp_pressure(temps, pressures, other_constants, "second")
    
    
    