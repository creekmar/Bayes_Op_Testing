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

National Institute of Standards and Technology. NIST. (2024, August 29). https://www.nist.gov/ 

Not above 80-85% of boiling point

printing speed : more closely parse <1mm/s region and a bit far away at >1mm/s region. 
maybe worth trying normalizing in logarithmic way.

Temperature Bounds where temperature min is 20 and max is based on .9 pressure or 140 (whatever is lower)
CF (20, 57.50681423147216)
TOL (20, 106.50003696848108)
CB (20, 127.37151737701168)
TCB (20, 140)
DCB (20, 140)

"""

from math import log
import matplotlib.pyplot as plt

CONSTANTS = {"CF": [4.56992, 1486.455, -8.612], "TOL": [4.08245, 1346.382, -53.508],
             "CB": [4.11083, 1435.675, -55.124], "TCB": [4.64002, 2110.983, -30.721],
             "DCB": [4.19518, 1649.55, -59.836]}
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
    f = make_pressure_from_temp_func(*CONSTANTS[solvent])
    low = f(to_Kelvin(temp_window[0]))
    high = f(to_Kelvin(temp_window[1]))
    if high > 1:
        high = 1
    return (low, high)

def get_all_solvent_bounds():
    """
    Get a dictionary of the solvnt temperature bounds relating 
    solvent name to solvent temperature bound in C
    """
    bound_dict = {}
    for solvent in CONSTANTS:
        b = calc_temp_bounds(solvent, (0, .9), (20, 140))
        bound_dict[solvent] = b
    return bound_dict


if __name__ == "__main__":
    # for solvent in CONSTANTS:
        # this is what we're using for the problem
        # print(solvent, calc_temp_bounds(solvent, (0, .9), (20, 140)))
        # print(solvent, ":", calc_pressure_bounds(solvent, (20, 140)))
    print(get_all_solvent_bounds())

    # constants = [4.20772, 1233.129, -40.953]
    # other_constants = [4.56992, 1486.455, -8.612]
    # temps = [425, 450, 475, 500, 525]
    # pressures = [10, 20, 30, 40, 50, 60]
    # plot_temp_pressure(temps, pressures, constants, "first")
    # plot_temp_pressure(temps, pressures, other_constants, "second")
    
    
    