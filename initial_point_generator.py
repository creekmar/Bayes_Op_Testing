"""
File: initial_point_generator.py
Author: Ming Creekmore
Purpose: Generate initial sobol points for lab automation problem and plot them using
         PCA and UMAP to show how well the points are spread out on the sample space
"""

import random
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skopt import sampler
import umap
from scipy.stats import gaussian_kde
from timeit import default_timer

from constants import get_all_press_solv_bounds, get_all_temp_solv_bounds, make_temp_from_pressure_f

# discrete input space, add D at the end to show it is dicrete
# for sampling, all sample spaces will be treated like categorical
# Should be in a list
SOLV_NAMES = ["CF", "CB", "CB9:A1", "CB8:A2", "CB7:A3"]
CONCEN_D = [5, 10]
PRINT_GAP_D = [50, 100]
PREC_VOL_D = [6, 9, 12]
MOTOR_SPEEDS_D = [0.01, 0.0355, 0.126, 0.4472, 1.587, 5.635, 20]
TEMP_CHOICES_D = {"CF": [25, 41.3], "CB": [25, 47.3, 62.9, 87.6, 107.4], "CB9:A1": [25, 47.3, 62.9, 87.6, 107.4],
                "CB8:A2": [25, 47.3, 62.9, 87.6, 107.4], "CB7:A3": [25, 47.3, 62.9, 87.6, 107.4]}
PRESSURE_CHOICES_D = {"CF": [0.258957, 0.5], "CB": [0.015971, 0.05, 0.1, 0.258957, 0.5],
                    "CB9:A1": [0.015971, 0.05, 0.1, 0.258957, 0.5], "CB8:A2": [0.015971, 0.05, 0.1, 0.258957, 0.5],
                    "CB7:A3": [0.015971, 0.05, 0.1, 0.258957, 0.5]}
LABELS_D = ["Motor Speed", "Temperature", "Concentration","Print Gap", "Precursor Volume", "Solvent"]

# Continuous Input Space, add C at the end to show it is continuous
# Should be a tuple to show low and high
# Should add decimal point to signify float
# Speed, Temp, vol, concentration, solvent
SPEED_C = (0.01, 20.0, 'log-uniform') # log-uniform is prior
TEMP_C = (25.0, 140.0)
PRESSURE_C = (0.0, 0.5)
PREC_VOL_C = (6.0, 12.0)
CONCEN_C = (1,5) # technically discrete from range 1 to 5. Will use math formula 2n-1 to correct to actual values
SPACE = [SPEED_C, PRESSURE_C, PREC_VOL_C, CONCEN_C, SOLV_NAMES]
SOLV_TEMP_BOUNDS = get_all_temp_solv_bounds(SOLV_NAMES, PRESSURE_C, TEMP_C)
SOLV_PRESS_BOUNDS = get_all_press_solv_bounds(SOLV_NAMES, TEMP_C, PRESSURE_C)
LABELS_C = ["Motor Speed", "Pressure", "Precursor Volume", "Concentration", "Solvent"]

SEED = 42

########################
# Discrete Data
########################

def get_all_discrete_points() -> set:
    """
    If the problem is discrete, this gets all points in the sample space
    [Speed, Temp, Concentration, print gap, vol, solvent]
    And returns it as a set
    """
    points = set()
    for s in SOLV_NAMES:
        for c in CONCEN_D:
            for g in PRINT_GAP_D:
                for v in PREC_VOL_D:
                    for m in MOTOR_SPEEDS_D:
                        for t in TEMP_CHOICES_D[s]:
                            points.add((m, t,c,g,v,s))
    return points

def discrete_sample_set(n: int, og_lists: list[list], sample: set = set()):
    """
    A half brute force way to pick samples in a discrete sample
    space, where each sample chosen is as distinct from the 
    previous samples chosen as possible
    @param n: number of points to generate
    @param og_lists: list of each discrete sample space
    @param sample: the set of samples that have already been picked. 
        Default will be a new set
    @return a list of points chosen
    """

    # s_lists are the "bags" we'll pick options from
    # c_lists are for the rebound if we accidentally pick a combination already in sample
    s_lists = [sublist[:] for sublist in og_lists]
    c_lists = [sublist[:] for sublist in og_lists]

    # max unique combinations
    max_num = 1
    for sublist in og_lists:
        max_num *= len(sublist)

    count = 0
    p_list = []

    while count < n:
        # clear sample lists if we've already picked all
        # available combinations
        if (len(sample) == max_num):
            print("CLEAR")
            sample.clear()
        point = []

        # make a semi-random point
        for i in range(len(og_lists)):
            sublist = s_lists[i]
            # restock
            if len(sublist) == 0:
                s_lists[i] = og_lists[i].copy()
                sublist = s_lists[i]
            c = sublist.pop(random.randint(0, len(sublist)-1))
            point.append(c)
        point = tuple(point)

        # add point if in sample
        if point not in sample:
            p_list.append(point)
            sample.add(point)
            count +=1
            for i in range(len(og_lists)):
                c_lists[i] = s_lists[i]
        else:
            # need to resample what we had before
            for i in range(len(og_lists)):
                s_lists[i] = c_lists[i]
            
    return p_list

def get_discrete_biased_initial_points(n=30, use_press = False):
    """
    Get evenly spaced points from the discrete sample space
    [Speed, Temp, Concentration, print gap, vol, solvent]
    @return a list of points
    """
    c_set = set()
    # getting points per solvent
    points = []
    num_solv = len(SOLV_NAMES)
    for s in range(num_solv):
        if (n//num_solv)*num_solv + s < n:  
            n_per_solv = (n//num_solv)+1
        else:
            n_per_solv = n//num_solv
        
        if use_press:
            choice_list = discrete_sample_set(n_per_solv, [MOTOR_SPEEDS_D, PRESSURE_CHOICES_D[SOLV_NAMES[s]], CONCEN_D, PRINT_GAP_D, PREC_VOL_D], c_set)
        else:
            choice_list = discrete_sample_set(n_per_solv, [MOTOR_SPEEDS_D, TEMP_CHOICES_D[SOLV_NAMES[s]], CONCEN_D, PRINT_GAP_D, PREC_VOL_D], c_set)
        points += [list(point) + [SOLV_NAMES[s]] for point in choice_list]

    return points


#################################
# Continuous Sample Space
#################################

def get_continuous_sobol_initial_points(n=14, use_press = False):
    """
    Gets sobol generated points in the space [Speed, pressure/temp, vol, concentration, solvent]
    where speed and temperature/press and vol are continuous
    The seed count is fixed so the same points will be generated each time
    Solvent will always have an even amount of points between each different solvent
    @param: n: number of points to generate
    @param: use_press: whether to use pressure or temperature
    @return: a list of points
    """
    
    count = 0
    points = []

    # initial point loop
    # will generate points randomly for each solvent in order
    # to keep the temperature bounds
    solv_num = len(SOLV_NAMES)
    for count in range(solv_num):
        # so we don't keep getting warnings, sobol sampler initialized here
        initial_point_gen = sampler.Sobol()
        if use_press:
            temp_bounds = SOLV_PRESS_BOUNDS[SOLV_NAMES[count]]
        else:
            temp_bounds = SOLV_TEMP_BOUNDS[SOLV_NAMES[count]]
        if (n//solv_num)*solv_num + count < n:  
            n_per_solv = (n//solv_num)+1
        else:
            n_per_solv = n//solv_num
        # need to have power of 2
        exponent = math.ceil(math.log2(n_per_solv))
        x = initial_point_gen.generate([SPEED_C,temp_bounds, PREC_VOL_C, CONCEN_C], 2**exponent, SEED+count)
        x = x[:n_per_solv]
        for p in x:
            # have to add the solvent 
            p.append(SOLV_NAMES[count])
            points.append(p)
        
    return points

def plot_pca_umap(df, model, graph, color, dataset_name, n_points, initial_counts = None, 
                  color_list = None, color_labels = None, num_levels = None):
    """
    model=DISCRETE vs CONTINUOUS
    graph = DENSITY vs SCATTER
    color = MONO vs MULTI
    dataset_name = "Lab_Automation"
    n_points = number of initial points vs sample space points
    num_levels = 10
    """
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(df)
   
    ##########
    # Get PCA
    ##########
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    X_pca = pca.fit_transform(X_standardized)
    pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])

    # plot
    fig = plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)

    if graph == "DENSITY":
        # Plot density of the sample pca data
        x_pca, y_pca = X_pca[:, 0], X_pca[:, 1]

        # Compute the KDE
        kde = gaussian_kde([x_pca, y_pca])

        # Create a grid of points for evaluation
        x_grid = np.linspace(x_pca.min() - 1, x_pca.max() + 1, 100)
        y_grid = np.linspace(y_pca.min() - 1, y_pca.max() + 1, 100)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
        Z = kde(positions).reshape(X_grid.shape)

        # plot density
        plt.contourf(X_grid, Y_grid, Z, levels=num_levels, cmap='Blues')
        plt.colorbar(label='Density')
        plt.title('PCA Contour Density Map')
    else:
        plt.scatter(pca_df['Principal Component 1'][n_points:], pca_df['Principal Component 2'][n_points:], c='grey', s=40) #, edgecolor='k')
        plt.title('PCA Scatterplot')

    # scatter plot
    if color == "MONO":
        plt.scatter(pca_df['Principal Component 1'][0:n_points], pca_df['Principal Component 2'][0:n_points], c='red', edgecolor='k', s=40)
    else:
        start = 0
        for i in range(len(initial_counts)):
            end = start + initial_counts[i] - 1
            plt.scatter(pca_df['Principal Component 1'][start:end], 
                        pca_df['Principal Component 2'][start:end], c=color_list[i], edgecolor='k', s=40)
            start += initial_counts[i]

    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')

    ##########
    # get UMAP
    ##########
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=1, n_jobs=1)
    umap_results = umap_model.fit_transform(X_standardized)
    umap_df = pd.DataFrame(data=umap_results, columns=['UMAP1', 'UMAP2'])

    # Plot density of the sample umap data
    plt.subplot(1, 2, 2)

    if graph == "DENSITY":
        # Extract UMAP components
        x_umap, y_umap = umap_results[:, 0], umap_results[:, 1]

        # Compute the KDE
        kde = gaussian_kde([x_umap, y_umap])

        # Get kde of the umap values
        x_grid = np.linspace(x_umap.min() - 1, x_umap.max() + 1, 100)
        y_grid = np.linspace(y_umap.min() - 1, y_umap.max() + 1, 100)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
        Z = kde(positions).reshape(X_grid.shape)

        plt.contourf(X_grid, Y_grid, Z, levels=num_levels, cmap='Blues')
        plt.colorbar(label='Density')
        plt.title('UMAP Contour Density Map')
    else:
        plt.scatter(umap_df['UMAP1'][n_points:], umap_df['UMAP2'][n_points:], c='grey', s=40) #, edgecolor='k')
        plt.title('UMAP Scatterplot')

    # scatter plot
    if color == "MONO":
        plt.scatter(umap_df['UMAP1'][0:n_points], umap_df['UMAP2'][0:n_points], c='red', edgecolor='k', s=40)
    else:
        start = 0
        for i in range(len(initial_counts)):
            end = start + initial_counts[i] - 1
            plt.scatter(umap_df['UMAP1'][start:end], 
                        umap_df['UMAP2'][start:end], c=color_list[i], edgecolor='k', s=40)
            start += initial_counts[i]

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    # adding legend for multicolors
    if color == "MULTI":
        # Create custom legend handles using color_list
        legend_handles = []
        for c in color_list:
            legend_handles.append(Line2D([0], [0], color=c, lw=2))
        if graph == "SCATTER":
            legend_handles.append(Line2D([0], [0], color='grey', lw=2))
            color_labels.append('SAMPLES')

        # Add the legend to the plot
        fig.legend(handles=legend_handles, labels=color_labels, loc=9, ncol=len(color_labels))

    # saving plot
    if graph == "DENSITY":
        plt.savefig(dataset_name + "_" + graph + "_" + model + "_" + str(num_levels) + "_" + color)
    else:
        plt.savefig(dataset_name + "_" + graph + "_" + model + "_" + color)
    plt.show()

if __name__ == "__main__":
    # n = 14
    # df = get_continuous_sobol_initial_points(n, use_press=True)
    # df.to_csv(str(n) + 'Points.csv')
    # print(df)
    # exit()

##############################
# Original plotting 
##############################

# """
    start = default_timer()
    n_points = 30
    discrete = False
    use_press = True
    reading_file = False

    # getting our sampling space
    if reading_file:
        df_sample = pd.read_csv(str(n_points) + "Points.csv").iloc[:,1:]
        df_sample = df_sample.sort_values('Solvent')
    else:
        if discrete:
            points = get_discrete_biased_initial_points(n_points, use_press)
            # NOTE: the column names will change depending on what the sample space is. Do make sure it is the right order
            df_sample = pd.DataFrame(points, columns=LABELS_D)
        else:
            points = get_continuous_sobol_initial_points(n_points, use_press)
            if use_press:
                df_sample = pd.DataFrame(points, columns=LABELS_C)

    our_points = df_sample.values.tolist()
    initial_counts = df_sample['Solvent'].value_counts().tolist()

    # Get points to represent the full sample space
    if discrete:
        space_points = get_all_discrete_points()
        # remove all the initial sample points in total points/ no overlapping
        for p in our_points:
            space_points.discard(tuple(p))
            
        df_space = pd.DataFrame(space_points, columns=LABELS_D)
        df_space = df_space.sort_values('Solvent')
        all_points = our_points+df_space.values.tolist()
        df_all = pd.DataFrame(all_points, columns=LABELS_D)
    else:
        # for continuous, we will just get a large amount of points to plot against
        space_points = get_continuous_sobol_initial_points(1000, use_press)
        df_space = pd.DataFrame(space_points, columns=LABELS_C)
        df_space = df_space.sort_values('Solvent')
        all_points = our_points+df_space.values.tolist()
        df_all = pd.DataFrame(all_points, columns=LABELS_C)

    # Replace solvent name with numbers for pca and umap
    df_all['Solvent'] = df_all['Solvent'].replace(SOLV_NAMES, [0,1,2,3,4])

    
    model="CONTINUOUS" # vs DISCRETE
    graph = "DENSITY" # vs SCATTER
    color = "MULTI" # vs MONO
    dataset_name = "Lab_Automation"
    color_list = ["orange", "red", "green", "cyan", "yellow"]
    color_labels = SOLV_NAMES.copy()
    color_labels.sort(key=str.lower)
    num_levels = 10
    plot_pca_umap(df_all, model, graph, color, dataset_name, n_points, initial_counts, color_list, color_labels, 10)

    # Save initial sobol points as they are to csv
    print(df_sample)
    df_sample["Motor Speed"] = df_sample["Motor Speed"].round(2)
    df_sample["Precursor Volume"] = df_sample["Precursor Volume"].round(1)
    df_sample.to_csv("initial_points_" + model + ".csv")

    # fixing values so they can be used in lab
    df_sample['Concentration'] = df_sample['Concentration'].multiply(2).subtract(1)
    # data was given in pressure, need to convert to temp
    for solv in SOLV_NAMES:
        f = make_temp_from_pressure_f(solv)
        df_sample.loc[df_sample['Solvent'] == solv, 'Temperature'] = df_sample.loc[df_sample['Solvent'] == solv, 'Pressure'].apply(f).subtract(273.15).round(2)
    df_sample.to_csv("fixed_initial_points_" + model + '.csv')
    print(df_sample)
    print("Time elapsed:", default_timer()- start)
# """