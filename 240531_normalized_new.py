import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from pyDOE import lhs
from dppy.finite_dpps import FiniteDPP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
from scipy.spatial.distance import pdist

# Define parameter ranges
solvents = [1, 2, 3, 4, 5]
# temperatures = [25, 50, 75, 100, 125, 150, 175, 200]
temperatures_1 = [20]
temperatures_2 = [20, 40, 60]
temperatures_3 = [20, 40, 60, 80, 100]
temperatures_4 = [20, 40, 60, 80, 100, 120, 140]
temperatures_5 = [20, 40, 60, 80, 100, 120, 140]
printing_speeds = [0.1, 0.5, 1, 5, 10, 15, 20, 25]
concentrations = [10, 15, 20]
printing_gaps = [25, 50, 75, 100]
precursor_volumes = [6, 9, 12]

# Generate all possible combinations of parameters
data = []
for solvent in solvents:
    if solvent == 1:
        temps = temperatures_1
    elif solvent == 2:
        temps = temperatures_2
    elif solvent == 3:
        temps = temperatures_3
    elif solvent == 4:
        temps = temperatures_4
    else:
        temps = temperatures_5

    for temp in temps:
        for speed in printing_speeds:
            for conc in concentrations:
                for gap in printing_gaps:
                    for precursor_volume in precursor_volumes:
                        data.append([solvent, temp, speed, conc, gap, precursor_volume])

data = np.array(data)

# Normalize parameter values
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Set a fixed random seed for k-means clustering
np.random.seed(50)

# Perform k-means clustering
n_clusters = 7  # Specify the desired number of clusters
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=1)
kmeans.fit(normalized_data)
labels = kmeans.labels_

# Select samples from each cluster and randomly select remaining points
kmeans_selected_indices = []
for i in range(n_clusters):
    cluster_points = normalized_data[labels == i]
    if len(cluster_points) > 0:
        kmeans_selected_indices.append(np.random.choice(np.where(labels == i)[0]))

remaining_points = 14 - len(kmeans_selected_indices)
if remaining_points > 0:
    remaining_indices = np.random.choice(np.delete(np.arange(normalized_data.shape[0]), kmeans_selected_indices), size=remaining_points, replace=False)
    kmeans_selected_indices.extend(remaining_indices)

kmeans_selected_points = normalized_data[kmeans_selected_indices]
kmeans_selected_data = data[kmeans_selected_indices]

# Set a fixed random seed for LHS
np.random.seed(50)

# Generate LHS sample
criterion = None
lhs_sample = lhs(6, samples=14, criterion=criterion)

lhs_selected_indices = []
for row in lhs_sample:
    solvent_index = int(round(row[0] * (len(solvents) - 1)))
    solvent = solvents[solvent_index]

    if solvent == 1:
        temps = temperatures_1
    elif solvent == 2:
        temps = temperatures_2
    elif solvent == 3:
        temps = temperatures_3
    elif solvent == 4:
        temps = temperatures_4
    else:
        temps = temperatures_5

    temp_index = int(round(row[1] * (len(temps) - 1)))
    speed_index = int(round(row[2] * (len(printing_speeds) - 1)))
    conc_index = int(round(row[3] * (len(concentrations) - 1)))
    gap_index = int(round(row[4] * (len(printing_gaps) - 1)))
    precursor_volume_index = int(round(row[5] * (len(precursor_volumes) - 1)))

    condition = (data[:, 0] == solvent) & (data[:, 1] == temps[temp_index]) & \
                (data[:, 2] == printing_speeds[speed_index]) & (data[:, 3] == concentrations[conc_index]) & \
                (data[:, 4] == printing_gaps[gap_index]) & (data[:, 5] == precursor_volumes[precursor_volume_index])

    matching_indices = np.where(condition)[0]
    if len(matching_indices) > 0:
        lhs_selected_indices.append(matching_indices[0])
    else:
        print(f"No matching point found for: {solvent}, {temps[temp_index]}, {printing_speeds[speed_index]}, {concentrations[conc_index]}, {printing_gaps[gap_index]}, {precursor_volumes[precursor_volume_index]}")

lhs_selected_data = data[lhs_selected_indices]
lhs_selected_points = normalized_data[lhs_selected_indices]


# Set a fixed random seed for k-DPP sampling
np.random.seed(50)

# Generate k-DPP sample
k = 14
dpp = FiniteDPP('likelihood', **{'L': np.eye(normalized_data.shape[0])})
dpp.sample_exact_k_dpp(k)
kdpp_selected_indices = dpp.list_of_samples[-1]
kdpp_selected_points = normalized_data[kdpp_selected_indices]
kdpp_selected_data = data[kdpp_selected_indices]

# Perform PCA on normalized data
pca = PCA(n_components=2)
pca_data = pca.fit_transform(normalized_data)
kmeans_pca_data = pca_data[kmeans_selected_indices]
lhs_pca_indices = np.array([np.argmin(np.sum((normalized_data - point) ** 2, axis=1)) for point in lhs_selected_points], dtype=int)
lhs_pca_data = pca_data[lhs_pca_indices]
kdpp_pca_data = pca_data[kdpp_selected_indices]

# Perform UMAP on normalized data
number_of_neighbors = 5
min_distance = 0.1
umap_data = umap.UMAP(n_neighbors=number_of_neighbors, min_dist=min_distance, metric='euclidean', random_state=1, n_jobs=1).fit_transform(normalized_data)
kmeans_umap_data = umap_data[kmeans_selected_indices]
lhs_umap_indices = np.array([np.argmin(np.sum((normalized_data - point) ** 2, axis=1)) for point in lhs_selected_points], dtype=int)
lhs_umap_data = umap_data[lhs_umap_indices]
kdpp_umap_data = umap_data[kdpp_selected_indices]

# Perform t-SNE on normalized data
tsne_perplexity = 10  # should be smaller than sample numbers
tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=1)
tsne_data = tsne.fit_transform(normalized_data)
kmeans_tsne_data = tsne_data[kmeans_selected_indices]
lhs_tsne_indices = np.array([np.argmin(np.sum((normalized_data - point) ** 2, axis=1)) for point in lhs_selected_points], dtype=int)
lhs_tsne_data = tsne_data[lhs_tsne_indices]
kdpp_tsne_data = tsne_data[kdpp_selected_indices]

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Define pastel colors for k-means data points
pastel_colors = sns.color_palette("pastel", n_colors=n_clusters)
kmeans_colors = [pastel_colors[i] for i in labels]

# PCA plots
axs[0, 0].scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_colors, alpha=0.3)
for i, point in enumerate(kmeans_pca_data):
    axs[0, 0].scatter(point[0], point[1], c='red', marker='x', s=100)
    axs[0, 0].text(point[0], point[1], str(i+1), fontsize=12, ha='center', va='center')
axs[0, 0].set_title('PCA - k-means')
legend_elements = [plt.Line2D([0], [0], marker='o', color=pastel_colors[i], label=f'Cluster {i+1}', linestyle='', alpha=0.8) for i in range(n_clusters)]
axs[0, 0].legend(handles=legend_elements, loc='lower right')

axs[0, 1].scatter(pca_data[:, 0], pca_data[:, 1], c='gray', alpha=0.3)
for i, point in enumerate(lhs_pca_data):
    axs[0, 1].scatter(point[0], point[1], c='red', marker='x', s=100)
    axs[0, 1].text(point[0], point[1], str(i+1), fontsize=12, ha='center', va='center')
axs[0, 1].set_title('PCA - LHS, ' + f'Criterion: {criterion}')

axs[0, 2].scatter(pca_data[:, 0], pca_data[:, 1], c='gray', alpha=0.3)
for i, point in enumerate(kdpp_pca_data):
    axs[0, 2].scatter(point[0], point[1], c='red', marker='x', s=100)
    axs[0, 2].text(point[0], point[1], str(i+1), fontsize=12, ha='center', va='center')
axs[0, 2].set_title('PCA - k-DPP')

# UMAP plots
axs[1, 0].scatter(umap_data[:, 0], umap_data[:, 1], c=kmeans_colors, alpha=0.3)
for i, point in enumerate(kmeans_umap_data):
    axs[1, 0].scatter(point[0], point[1], c='red', marker='x', s=100)
    axs[1, 0].text(point[0], point[1], str(i+1), fontsize=12, ha='center', va='center')
axs[1, 0].set_title('UMAP - k-means')
legend_elements = [plt.Line2D([0], [0], marker='o', color=pastel_colors[i], label=f'Cluster {i+1}', linestyle='', alpha=0.8) for i in range(n_clusters)]
axs[1, 0].legend(handles=legend_elements, loc='lower right')
axs[1, 0].text(0.95, 0.95, f'n_neighbors={number_of_neighbors}, \nmin_dist={min_distance}', transform=axs[1, 0].transAxes, fontsize=10, va='top', ha='right')

axs[1, 1].scatter(umap_data[:, 0], umap_data[:, 1], c='gray', alpha=0.3)
for i, point in enumerate(lhs_umap_data):
    axs[1, 1].scatter(point[0], point[1], c='red', marker='x', s=100)
    axs[1, 1].text(point[0], point[1], str(i+1), fontsize=12, ha='center', va='center')
axs[1, 1].set_title('UMAP - LHS, ' + f'Criterion: {criterion}')
axs[1, 1].text(0.95, 0.95, f'n_neighbors={number_of_neighbors}, \nmin_dist={min_distance}', transform=axs[1, 1].transAxes, fontsize=10, va='top', ha='right')

axs[1, 2].scatter(umap_data[:, 0], umap_data[:, 1], c='gray', alpha=0.3)
for i, point in enumerate(kdpp_umap_data):
    axs[1, 2].scatter(point[0], point[1], c='red', marker='x', s=100)
    axs[1, 2].text(point[0], point[1], str(i+1), fontsize=12, ha='center', va='center')
axs[1, 2].set_title('UMAP - k-DPP')
axs[1, 2].text(0.95, 0.95, f'n_neighbors={number_of_neighbors}, \nmin_dist={min_distance}', transform=axs[1, 2].transAxes, fontsize=10, va='top', ha='right')


# t-SNE plots
axs[2, 0].scatter(tsne_data[:, 0], tsne_data[:, 1], c=kmeans_colors, alpha=0.3)
for i, point in enumerate(kmeans_tsne_data):
    axs[2, 0].scatter(point[0], point[1], c='red', marker='x', s=100)
    axs[2, 0].text(point[0], point[1], str(i+1), fontsize=12, ha='center', va='center')
axs[2, 0].set_title('t-SNE - k-means')
legend_elements = [plt.Line2D([0], [0], marker='o', color=pastel_colors[i], label=f'Cluster {i+1}', linestyle='', alpha=0.8) for i in range(n_clusters)]
axs[2, 0].legend(handles=legend_elements, loc='lower right')
axs[2, 0].text(0.95, 0.95, f'perplexity={tsne_perplexity}', transform=axs[2, 0].transAxes, fontsize=10, va='top', ha='right')

axs[2, 1].scatter(tsne_data[:, 0], tsne_data[:, 1], c='gray', alpha=0.3)
for i, point in enumerate(lhs_tsne_data):
    axs[2, 1].scatter(point[0], point[1], c='red', marker='x', s=100)
    axs[2, 1].text(point[0], point[1], str(i+1), fontsize=12, ha='center', va='center')
axs[2, 1].set_title('t-SNE - LHS, ' + f'Criterion: {criterion}')
axs[2, 1].text(0.95, 0.95, f'perplexity={tsne_perplexity}', transform=axs[2, 1].transAxes, fontsize=10, va='top', ha='right')

axs[2, 2].scatter(tsne_data[:, 0], tsne_data[:, 1], c='gray', alpha=0.3)
for i, point in enumerate(kdpp_tsne_data):
    axs[2, 2].scatter(point[0], point[1], c='red', marker='x', s=100)
    axs[2, 2].text(point[0], point[1], str(i+1), fontsize=12, ha='center', va='center')
axs[2, 2].set_title('t-SNE - k-DPP')
axs[2, 2].text(0.95, 0.95, f'perplexity={tsne_perplexity}', transform=axs[2, 2].transAxes, fontsize=10, va='top', ha='right')


plt.tight_layout()
plt.show()

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print selected data points in table format
kmeans_df = pd.DataFrame(kmeans_selected_data, columns=['Solvent', 'Temperature', 'Printing Speed', 'Concentration', 'Gap', 'Precursor_Volume'])
lhs_df = pd.DataFrame(lhs_selected_data, columns=['Solvent', 'Temperature', 'Printing Speed', 'Concentration', 'Gap', 'Precursor_Volume'])
kdpp_df = pd.DataFrame(kdpp_selected_data, columns=['Solvent', 'Temperature', 'Printing Speed', 'Concentration', 'Gap', 'Precursor_Volume'])

print("K-means Selected Data Points:")
print(kmeans_df)
print("\nLHS Selected Data Points:")
print(lhs_df)
print("\nk-DPP Selected Data Points:")
print(kdpp_df)

# Reset pandas display options to default
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# Calculate mean Euclidean distances
def mean_euclidean_distance(points):
    distances = pdist(points, metric='euclidean')
    return np.mean(distances)

kmeans_mean_distance = mean_euclidean_distance(kmeans_selected_data)
lhs_mean_distance = mean_euclidean_distance(lhs_selected_data)
kdpp_mean_distance = mean_euclidean_distance(kdpp_selected_data)

print("Mean Euclidean Distance:")
print(f"k-means: {kmeans_mean_distance:.4f}")
print(f"LHS: {lhs_mean_distance:.4f}")
print(f"k-DPP: {kdpp_mean_distance:.4f}")