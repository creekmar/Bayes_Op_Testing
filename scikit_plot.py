"""
File: scikit_plot.py
Author: Ming Creekmore
Purpose: Some helper functions to plot standardized specific results for data testing
"""

import random
from matplotlib import cm, pyplot as plt, colors
from matplotlib.axes import Axes
import numpy as np
from skopt.plots import plot_convergence, plot_objective
import pandas as pd

MATERIAL_TEMP = [("CU", 25.0), ("TP", 50.0), ("MN", 70.0), ("SN", 90.0), ("PC", 40.0), ("DR", 40.0)]

def plot_true_objective(f_range, f, params = None, n_pts = 200, directory="./"):
    """
    Plots a function using matplotlib. We are assuming that it is only 3 dimensional
    @param: f_range - ((x_min, x_max), (y_min, y_max)), showing the range of x and y range you want to graph
    @param: f - the function to plot, we assume that the function has at least one argument to take in (x,y) 
                coordinate to plot
    @param: params - the extra params that the function needs
    @param: n_points - the number of points to plot
    """

    ##### Plot the objective function without noise for our reference (Can't do this for real experiements)   
    x1 = np.linspace(f_range[0][0], f_range[0][1], n_pts).reshape(-1, 1)
    x2 = np.linspace(f_range[1][0], f_range[1][1], n_pts).reshape(-1, 1)
    Y = np.zeros((n_pts, n_pts))
    for i in range(len(x1)):
        for j in range(len(x2)):
            if params is None:
                Y[j,i] = f([x1[i], x2[j]])
            else:
                Y[j,i] = f([x1[i], x2[j]], *params)

    X1, X2 = np.meshgrid(x1, x2)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm)

    ax.set_xlabel('Temperature')
    ax.set_ylabel('Speed')
    ax.set_zlabel('Data')
    plt.title('True Objective Function')
    fig1.colorbar(surf, shrink=0.5, aspect=5)
    plt.ion()
    plt.show() 
    plt.savefig(directory + "true_objective")

def plt4d(f, data=None, n=100, directory="./"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if data is None:
        z = np.random.randint(0, 6, n)
        x = np.zeros(n)
        y = np.zeros(n)
        for i in range(n):
            x[i] = (random.random() * 45) + 5 # motor
            sd = 20
            bp = MATERIAL_TEMP[z[i]][1]
            if 100-bp<sd:
                sd = 100-bp
            y[i] = (random.random() * sd) + bp # temperature
    else:
        x, y, z = data

    c = np.zeros(n)
    for i in range(n):
        c[i] = f([x[i], y[i], z[i]])

    img = ax.scatter(x, y, c, c=z, cmap=plt.hot())
    fig.colorbar(img)
    plt.savefig(directory + "4d_objetive")
    plt.show()
    df = pd.DataFrame({'motor': x, 'heater': y, 'material': z, 'objective': c}, 
                      columns=['motor', 'heater', 'material', 'objective'])
    print(df)
    df.to_csv("4d_objective.csv")

def plot_optimization_trace(iterations, objective_values):
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, objective_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Optimization Trace')
    plt.grid(True)
    plt.savefig("Optimization_Trace")

def print_results(result, directory = "./"):
    """
    Print the testing results of scikit-optimize. Save convergence and objective plot
    @param: result - Scikit Optimizer result after all iterations and testing are done
    @param: directory - The directory to save the figures in
    """
    # Printing results
    print("========== RESULT ==========")
    print(result)
    print()
    print("Lowest minimum of " + str(result.fun) + " at " + str(result.x))
    fig2 = plt.figure()
    plot_convergence(result)
    # input("Press enter to see fig 1")
    
    plt.savefig(directory + "convergence_plot")
    plt.show()
    # fig3 = plt.figure()
    plot_objective(result)
    # input("Press enter to see fig 2")
    plt.savefig(directory + "objective_plot")
    plt.show()