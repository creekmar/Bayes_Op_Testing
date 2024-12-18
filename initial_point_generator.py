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

from constants import make_temp_from_pressure_f, get_all_press_solv_bounds

# discrete/categorical input space
SOLV_NAMES = ["CF", "CB", "CB:A1", "CB:A2", "CB:A3"]
BP = [61.2, 131.2, 133.4, 135.6, 137.8]
CONCEN = [5, 10]
PRINT_GAP = [50, 100]
PREC_VOL = [6, 9, 12]
MOTOR_SPEEDS = [0.01, 0.0355, 0.126, 0.4472, 1.587, 5.635, 20]
TEMP_CHOICES = {"CF": [25, 41.3], "CB": [25, 47.3, 62.9, 87.6, 107.4], "CB:A1": [25, 47.3, 62.9, 87.6, 107.4],
                "CB:A2": [25, 47.3, 62.9, 87.6, 107.4], "CB:A3": [25, 47.3, 62.9, 87.6, 107.4]}
PRESSURE_CHOICES = {"CF": [0.258957, 0.5], "CB": [0.015971, 0.05, 0.1, 0.258957, 0.5],
                    "CB:A1": [0.015971, 0.05, 0.1, 0.258957, 0.5], "CB:A2": [0.015971, 0.05, 0.1, 0.258957, 0.5],
                    "CB:A3": [0.015971, 0.05, 0.1, 0.258957, 0.5]}
SEED = 42

# Speed, Temp, Concentration, print gap, vol, solvent
SPACE = [(0.01,20.0),(25.0,140.0), CONCEN, PRINT_GAP, PREC_VOL, BP]
SOLV_TEMP_BOUNDS = {'CF': (25, 41.2), 'CB': (25, 112), 'CB:A1': (25, 112), 'CB:A2': (25, 112), 'CB:A3': (25, 112)}
SOLV_PRESS_BOUNDS = get_all_press_solv_bounds(25, 140)

def get_all_discrete_points() -> set:
    """
    If the problem is discrete, this gets all points in the sample space
    [Speed, Temp, Concentration, print gap, vol, solvent]
    And returns it as a set
    """
    points = set()
    for s in SOLV_NAMES:
        for c in CONCEN:
            for g in PRINT_GAP:
                for v in PREC_VOL:
                    for m in MOTOR_SPEEDS:
                        for t in TEMP_CHOICES[s]:
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

def get_discrete_biased_initial_points(n=30):
    """
    Get evenly spaced points from the discrete sample space
    [Speed, Temp, Concentration, print gap, vol, solvent]
    @return a dataframe of the points
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
        
        choice_list = discrete_sample_set(n_per_solv, [MOTOR_SPEEDS, TEMP_CHOICES[SOLV_NAMES[s]], CONCEN, PRINT_GAP, PREC_VOL], c_set)
        points += [list(point) + [s] for point in choice_list]
    
    # fixing dataframe of data to be rounded and solvent names to replace numbers
    df = pd.DataFrame(points, columns=["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Solvent"])
    df['Solvent'] = df['Solvent'].replace([0,1,2,3,4], SOLV_NAMES)
    return df


#################################
# Speed and Temp are Continuous #
#################################

def generate_logarithmic_data(start, end, num_points, random = False):
    """
    Generates logarithmic spaced points between start and end"""
    if random: 
        random_log_values = np.random.uniform(np.log10(start), np.log10(end), num_points)
        data_points = 10 ** random_log_values
    else:
        # generate evenly spaced points
        data_points = np.logspace(np.log10(start), np.log10(end), num=num_points)
    return data_points

def get_sample_space_points(brute_force = True, n_points = 500, m_increment = 2, t_increment = 10):
    """
    This is a brute force approach to show the data points of all points in the sample space
    [Speed, Temp, Concentration, print gap, vol, solvent]
    where speed and temp are continous
    To reduce the time, increase the motor and temperature increments so there are less points
    To get more points, decrease the motor and temperature increments
    @param: brute force: boolean to determine whether to partisan it ourselves or to draw random samples
    @param: n_points: number of points to generate. Only used when brute_force=False
    @param: m_increment: the motor speed increment to partisan by when brute_force=True
    @param: t_increment: the temperature increment to partisan by when brute_force=True
    @return: A list of points in space [Speed, Temp, Concentration, print gap, vol, solvent]
             theoretically to represent the whole sample space
    """

    count = 0
    points = []
    if brute_force:
        # # the solvent
        for s in range(len(SOLV_NAMES)):
            # prec vol
            for v in range(len(PREC_VOL)):
                # print gap
                for g in range(len(PRINT_GAP)):
                    # concentration
                    for c in range(len(CONCEN)):
                        temp_bounds = SOLV_TEMP_BOUNDS[SOLV_NAMES[s]]
                        t = temp_bounds[0]
                        while t <= temp_bounds[1]:
                            m = 0.1
                            while m <= 25.0:
                                points.append([m, t, CONCEN[c], PRINT_GAP[g], PREC_VOL[v], s])
                                m += m_increment
                            points.append([25.0, t, CONCEN[c], PRINT_GAP[g], PREC_VOL[v], s])
                            t += t_increment

    else: # just get random data sample
        for count in range(5):
            initial_point_gen = sampler.Lhs()
            solv = count%5
            temp_bounds = SOLV_TEMP_BOUNDS[SOLV_NAMES[solv]]
            x = initial_point_gen.generate([(0.1,25.0),temp_bounds, CONCEN, PRINT_GAP, PREC_VOL], n_points,SEED+count)
            for p in x:
                # have to add the solvent 
                p.append(BP[solv])
                points.append(p)
    return points

def get_continuous_sobol_initial_points(n=14, use_BP=False, use_press = False):
    """
    Gets sobol generated points in the space [Speed, Temp, Concentration, print gap, vol, solvent]
    where speed and temperature are continuous
    The seed count is fixed so the same points will be generated each time
    Solvent will always have an even amount of points between each different solvent
    @param: n: number of points to generate
    @param: use_BP: whether to replace solvent number with its boiling points
    @param: use_press: whether to use pressure or temperature
    @return: points in a list 
    """
    
    count = 0
    points = []

    # initial point loop
    # will generate points randomly for each solvent in order
    # to keep the temperature bounds
    for count in range(5):
        # so we don't keep getting warnings, sobol sampler initialized here
        initial_point_gen = sampler.Sobol()
        solv = count%5
        if use_press:
            temp_bounds = SOLV_PRESS_BOUNDS[SOLV_NAMES[solv]]
        else:
            temp_bounds = SOLV_TEMP_BOUNDS[SOLV_NAMES[solv]]
        if (n//5)*5 + count < n:  
            n_per_solv = (n//5)+1
        else:
            n_per_solv = n//5
        # need to have power of 2
        exponent = math.ceil(math.log2(n_per_solv))
        x = initial_point_gen.generate([(0.1,25.0),temp_bounds, CONCEN, PRINT_GAP, PREC_VOL], 2**exponent, SEED+count)
        x = x[:n_per_solv]
        for p in x:
            # have to add the solvent 
            if use_BP:
                p.append(BP[solv])
            else:
                p.append(solv)
            points.append(p)
        
    return points

def get_continous_biased_initial_points(n=30, use_BP=False, use_press = False):
    """
    Gets semi-biased generated points in the space 
    [Speed, Temp, Concentration, print gap, vol, solvent]
    where speed and temperature are continuous 
    The speed is uniformly distributed, while the temperature
    is evenly spaced logarithmically by pressure
    The concentration, print gap, vol, and solvent are chosen randomly
    @param: n: number of points to generate
    @param: use_BP: whether to replace solvent number with its boiling points
    @param: use_press: whether to use pressure or temperature
    @return: points in a list 
    """
    # get all combinations of volume, gap, and concentration
    # prec vol
    combinations = []
    for v in range(3):
        # print gap
        for g in range(4):
            # concentration
            for c in range(3):     
                combinations.append([CONCEN[c], PRINT_GAP[g], PREC_VOL[v]])
    
    choice_list = np.random.choice(len(combinations), n, replace=False)
    choice_count = 0
    # getting points per solvent
    points = []
    for s in range(5):
        if (n//5)*5 + s < n:  
            n_per_solv = (n//5)+1
        else:
            n_per_solv = n//5
        # we want motor speeds more forcused on the lower values
        m_array = generate_logarithmic_data(0.01, 20, n_per_solv)
        # we want temperatures evenly spaced by the pressure bounds
        bounds = SOLV_PRESS_BOUNDS[SOLV_NAMES[s]]
        t_array = np.random.uniform(bounds[0], bounds[1], n_per_solv)

        count = 0
        while count < n_per_solv:
           
            points.append([m_array[count]] + [t_array[count]] + combinations[choice_list[choice_count]] + [s])
            choice_count+=1
            count+=1

    # fixing dataframe of data to be rounded and solvent names to replace numbers
    df = pd.DataFrame(points, columns=["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Solvent"])
    # df = pd.DataFrame(points, columns=["Solvent", "Pressure"])
    df["Motor Speed"] = df["Motor Speed"].round(2)
    if not use_BP:
        df['Solvent'] = df['Solvent'].replace([0,1,2,3,4], SOLV_NAMES)

    # data was given in pressure, need to convert to temp
    if not use_press:
        for solv in SOLV_NAMES:
            f = make_temp_from_pressure_f(solv)
            df.loc[df['Solvent'] == solv, 'Temperature'] = df.loc[df['Solvent'] == solv, 'Temperature'].apply(f).subtract(273.15)
    df['Temperature'] = df['Temperature'].round(1)
    return df

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
    # n = 70
    # df = get_discrete_biased_initial_points(n)
    # df.to_csv(str(n) + 'Points.csv')
    # print(df)
    # exit()

##############################
# Original plotting 
##############################

# """
    start = default_timer()
    n_points = 70
    # our_points = get_sobol_initial_points(n_points, use_press=True)
    df = pd.read_csv(str(n_points) + "Points.csv").iloc[:,1:]
    df = df.sort_values('Solvent')
    initial_counts = df['Solvent'].value_counts().tolist()
    our_points = df.values.tolist()

    # we want to get all available points in the space
    # space_points = get_sample_space_points(True, 500, 4, 20)
    space_points = get_all_discrete_points()

    # remove all the initial sample points in total points
    for p in our_points:
        space_points.discard(tuple(p))

    df = pd.DataFrame(space_points, columns=["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Solvent"])
    df = df.sort_values('Solvent')
    # sample_counts = df["Solvent"].value_counts().tolist()
    all_points = our_points+df.values.tolist()


    # round and standardize all points
    df = pd.DataFrame(all_points, columns=["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Solvent"])
    df['Solvent'] = df['Solvent'].replace(SOLV_NAMES, [0,1,2,3,4])
    # df["Motor Speed"] = df["Motor Speed"].round(1)
    # df["Temperature"] = df["Temperature"].round(1)

    # Save initial sobol points to csv
    # df['Solvent'] = df['Solvent'].replace([0,1,2,3,4], SOLV_NAMES)
    # df.iloc[:n_points].to_csv("initial_points_" + model + ".csv")
    
    model="DISCRETE" # vs CONTINUOUS
    graph = "DENSITY" # vs SCATTER
    color = "MONO" # vs MULTI
    dataset_name = "Lab_Automation"
    color_list = ["orange", "red", "green", "cyan", "yellow"]
    color_labels = SOLV_NAMES.copy()
    color_labels.sort(key=str.lower)
    num_levels = 10
    plot_pca_umap(df, "DISCRETE", "SCATTER", "MULTI", "Lab_Automation", n_points, initial_counts, color_list, color_labels, 10)

    print("Time elapsed:", default_timer()- start)
# """