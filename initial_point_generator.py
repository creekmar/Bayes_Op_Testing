"""
File: initial_point_generator.py
Author: Ming Creekmore
Purpose: Generate initial sobol points for lab automation problem and plot them using
         PCA and UMAP to show how well the points are spread out on the sample space
"""

from matplotlib import pyplot as plt
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
SOLV_NAMES = ["CF", "TOL", "CB", "mXY", "MES"]
BP = [61.2, 110.6, 132, 174.1, 180.1]
CONCEN = [10, 15, 20]
PRINT_GAP = [25, 50, 75, 100]
PREC_VOL = [6, 9, 12]
MOTOR_SPEEDS_9 = [0.01, 0.02586, 0.066874, 0.17, 0.44721, 1.1565, 2.99, 7.734, 20]
MOTOR_SPEEDS_10 = [0.01, 0.02327, 0.054145, 0.126, 0.2932, 0.6822, 1.5874, 3.69375, 8.595, 20]
TEMP_CHOICES = {"CF": [25,41.29986747], "TOL": [25,30.45310597,45.26606599,68.7105251,87.50711471], 
                    "CB": [25,47.25710066,62.88238474,87.59268613,107.3866381], 
                    "mXY": [25,53.95836057,69.73158755,94.65825633,114.6099936],
                    "MES": [25,75.79225276,92.31476572,118.3802477,139.2036371]}
PRESSURE_CHOICES = {"CF": [0.25895711, 0.5], "TOL": [0.037929005, 0.05, 0.1, 0.25895711, 0.5], 
                    "CB": [0.015971088, 0.05, 0.1, 0.25895711, 0.5], 
                    "mXY": [0.011050063, 0.05, 0.1, 0.25895711, 0.5],
                    "MES": [0.003221155, 0.05, 0.1, 0.25895711, 0.5]}
SEED = 42

# Speed, Temp, Concentration, print gap, vol, solvent
SPACE = [(0.1,20.0),(25.0,140.0), CONCEN, PRINT_GAP, PREC_VOL, BP]
SOLV_TEMP_BOUNDS = {'CF': (25, 41.2), 'TOL': (25, 90.6), 'CB': (25, 112), 'DEC': (25, 140),'DCB': (25, 140)}
SOLV_PRESS_BOUNDS = get_all_press_solv_bounds(25, 140)


def generate_logarithmic_data(start, end, num_points, random = False):
    """Generates logarithmic spaced points between start and end"""
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
        for s in range(5):
            # prec vol
            for v in range(3):
                # print gap
                for g in range(4):
                    # concentration
                    for c in range(3):
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
            x = initial_point_gen.generate([(0.1,25.0),temp_bounds, CONCEN, PRINT_GAP, PREC_VOL], n_points,SEED)
            for p in x:
                # have to add the solvent 
                p.append(BP[solv])
                points.append(p)
    return points

def get_sobol_initial_points(n=14, use_BP=False, use_press = False):
    """
    Gets sobol generated points in the space [Speed, Temp, Concentration, print gap, vol, solvent]
    The seed count is fixed so the same points will be generated each time
    Solvent will always have an even amount of points between each different solvent
    @param: n: number of points to generate
    @param: use_BP: whether to replace solvent number with its boiling points
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

def get_biased_initial_points(n=30):
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
    # and calculating temperature to replace pressure
    df = pd.DataFrame(points, columns=["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Solvent"])
    # df = pd.DataFrame(points, columns=["Solvent", "Pressure"])
    df["Motor Speed"] = df["Motor Speed"].round(2)
    df['Solvent'] = df['Solvent'].replace([0,1,2,3,4], SOLV_NAMES)
    for solv in SOLV_NAMES:
        f = make_temp_from_pressure_f(solv)
        df.loc[df['Solvent'] == solv, 'Temperature'] = df.loc[df['Solvent'] == solv, 'Temperature'].apply(f).subtract(273.15)
    df['Temperature'] = df['Temperature'].round(1)
    return df

if __name__ == "__main__":
    df = get_biased_initial_points()
    df.to_csv('30Points.csv')
    print(df)
    exit()

##############################
# Original plotting 
##############################

"""
    start = default_timer()
    n_points = 30
    our_points = get_sobol_initial_points(n_points, use_press=True)
    model="SOBOL"

    # TEST
    df = pd.DataFrame(our_points, columns=["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Solvent"])
    df["Motor Speed"] = df["Motor Speed"].round(1)
    df['Solvent'] = df['Solvent'].replace([0,1,2,3,4], SOLV_NAMES)
    
    for solv in SOLV_NAMES:
        f = make_temp_from_pressure_f(solv)
        df.loc[df['Solvent'] == solv, 'Temperature'] = df.loc[df['Solvent'] == solv, 'Temperature'].apply(f).subtract(273.15)
        
    print(df)
    # print(df)
    # we want <10k data points
    # space_points = get_sample_space_points(True, 500, 4, 20)
    # print(len(space_points))
    df.to_csv("30points.csv")
    exit()

    all_points = our_points+space_points

    # round and standardize all points
    df = pd.DataFrame(all_points, columns=["Motor Speed", "Temperature", "Concentration", "Printing Gap", "Precursor Volume", "Solvent"])
    df["Motor Speed"] = df["Motor Speed"].round(1)
    # df["Temperature"] = df["Temperature"].round(1)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(df)

    # Save initial sobol points to csv
    df['Solvent'] = df['Solvent'].replace([0,1,2,3,4], SOLV_NAMES)
    df.iloc[:n_points].to_csv("initial_points_" + model + ".csv")
   
    # Get PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    X_pca = pca.fit_transform(X_standardized)

    dataset_name = "Lab_Automation"

    # plot
    plt.figure(figsize=(14, 6))

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

    kde = gaussian_kde(X_pca.T)
    density = kde(X_pca.T)
    plt.subplot(1, 2, 1)
    # plt.tricontourf(X_pca[:, 0], X_pca[:, 1], density, cmap='Blues')
    plt.contourf(X_grid, Y_grid, Z, levels=20, cmap='Blues')
    plt.colorbar(label='Density')
    # scatter plot of our points over the density
    pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
    plt.scatter(pca_df['Principal Component 1'][0:n_points], pca_df['Principal Component 2'][0:n_points], c='red', edgecolor='k', s=40)
    plt.title('PCA Contour Density Map')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    # plt.savefig("PCA_" + dataset_name + "_" + model)
    # plt.show()
    # plt.cla()
    # exit()

    # get UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=1, n_jobs=1)
    umap_results = umap_model.fit_transform(X_standardized)

    # Plot density of the sample umap data

    # Extract UMAP components
    x_umap, y_umap = umap_results[:, 0], umap_results[:, 1]

    # Compute the KDE
    kde = gaussian_kde([x_umap, y_umap])

    # Create a grid of points for evaluation
    x_grid = np.linspace(x_umap.min() - 1, x_umap.max() + 1, 100)
    y_grid = np.linspace(y_umap.min() - 1, y_umap.max() + 1, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    Z = kde(positions).reshape(X_grid.shape)

    kde = gaussian_kde(umap_results.T)
    density = kde(umap_results.T)
    plt.subplot(1, 2, 2)
    # plt.tricontourf(umap_results[:, 0], umap_results[:, 1], density, cmap='Blues')
    plt.contourf(X_grid, Y_grid, Z, levels=20, cmap='Blues')
    plt.colorbar(label='Density')
    # scatter plot of our points over the density
    umap_df = pd.DataFrame(data=umap_results, columns=['UMAP1', 'UMAP2'])
    plt.scatter(umap_df['UMAP1'][0:n_points], umap_df['UMAP2'][0:n_points], c='red', edgecolor='k', s=40)
    plt.title('UMAP Contour Density Map')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig("SCATTER_" + dataset_name + "_" + model)
    plt.show()

    print("Time elapsed:", default_timer()- start)
# """