import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from minisom import MiniSom
import os

# Constants
OUTPUT_FOLDER = 'Output'
DATA_FILE_PATH = 'Data/sp500_final_2023.csv'

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class StockClustering:
    def __init__(self, algorithms):
        self.algorithms = algorithms

    @staticmethod
    def load_data(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def preprocess_data(data):
        """
        Preprocess data and apply automated feature selection techniques.
        """
        features = data[['Daily Return_mean', 'Daily Return_std', 'Cumulative Return_mean', '30 Day MA_mean',
                         '30 Day STD_mean', '30 Day EMA_mean', 'RSI_mean', 'Bollinger High_mean', 'Bollinger Low_mean',
                         'Volume_mean']]
        
        # Apply Variance Threshold
        selector = VarianceThreshold(threshold=0.01)
        features_var = selector.fit_transform(features)
        selected_features = features.columns[selector.get_support()]

        # Apply RFE
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(model, n_features_to_select=5)
        rfe.fit(features_var, data['Cluster'])
        selected_features_rfe = selected_features[rfe.support_]

        # Return the selected features
        return data[selected_features_rfe]

    @staticmethod
    def standardize_features(features):
        scaler = StandardScaler()
        return scaler.fit_transform(features), scaler

    @staticmethod
    def apply_pca(scaled_features, n_components=2):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(scaled_features), pca

    @staticmethod
    def plot_pca_clusters(principal_components, clusters, output_folder, title):
        pca_data = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
        pca_data['Cluster'] = clusters

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Cluster', palette='viridis', data=pca_data)
        plt.title(title)
        plt.savefig(os.path.join(output_folder, f'pca_clusters_{title.lower().replace(" ", "_")}.png'))
        plt.show()

    def evaluate_and_save(self, scaled_features, clusters, file_suffix):
        silhouette_avg = silhouette_score(scaled_features, clusters)
        davies_bouldin = davies_bouldin_score(scaled_features, clusters)
        calinski_harabasz = calinski_harabasz_score(scaled_features, clusters)

        print(f'Silhouette Score: {silhouette_avg}')
        print(f'Davies-Bouldin Index: {davies_bouldin}')
        print(f'Calinski-Harabasz Index: {calinski_harabasz}')

        print(f'PCA cluster plot saved to {os.path.join(OUTPUT_FOLDER, f"pca_clusters_{file_suffix}.png")}')

    def run(self, data_file):
        data = self.load_data(data_file)

        # Assuming 'Cluster' column is already present for RFE
        if 'Cluster' not in data.columns:
            kmeans = KMeans(n_clusters=5, random_state=42)
            data['Cluster'] = kmeans.fit_predict(data[['Daily Return_mean', 'Daily Return_std', 'Cumulative Return_mean', '30 Day MA_mean',
                                                       '30 Day STD_mean', '30 Day EMA_mean', 'RSI_mean', 'Bollinger High_mean', 'Bollinger Low_mean',
                                                       'Volume_mean']])

        features = self.preprocess_data(data)
        scaled_features, scaler = self.standardize_features(features)

        results_df = data.copy()

        for algorithm in self.algorithms:
            clusters, file_suffix, title = algorithm(scaled_features)
            results_df[f'Cluster_{file_suffix}'] = clusters

            principal_components, pca = self.apply_pca(scaled_features)
            self.plot_pca_clusters(principal_components, clusters, OUTPUT_FOLDER, title)
            self.evaluate_and_save(scaled_features, clusters, file_suffix)

        results_df.to_csv(os.path.join(OUTPUT_FOLDER, 'sp500_clustered_results_2023.csv'), index=False)
        print(f'All cluster results saved to {os.path.join(OUTPUT_FOLDER, "sp500_clustered_results_2023.csv")}')

def som_algorithm(scaled_features):
    x_range = [5, 10, 15, 20]
    y_range = [5, 10, 15, 20]
    sigma_range = [0.5, 1.0, 1.5, 2.0]
    learning_rate_range = [0.1, 0.5, 0.9]

    best_params = {'x': 10, 'y': 10, 'sigma': 1.0, 'learning_rate': 0.5}
    best_silhouette = -1

    for x in x_range:
        for y in y_range:
            for sigma in sigma_range:
                for lr in learning_rate_range:
                    som = MiniSom(x=x, y=y, input_len=scaled_features.shape[1], sigma=sigma, learning_rate=lr)
                    som.random_weights_init(scaled_features)
                    som.train_random(scaled_features, 100)
                    win_map = som.win_map(scaled_features)
                    clusters = np.zeros(len(scaled_features))
                    for cluster, points in win_map.items():
                        for point in points:
                            idx = np.where((scaled_features == point).all(axis=1))[0][0]
                            clusters[idx] = cluster[0] * som.get_weights().shape[1] + cluster[1]
                    if len(set(clusters)) > 1:
                        silhouette_avg = silhouette_score(scaled_features, clusters)
                        if silhouette_avg > best_silhouette:
                            best_silhouette = silhouette_avg
                            best_params = {'x': x, 'y': y, 'sigma': sigma, 'learning_rate': lr}

    som = MiniSom(x=best_params['x'], y=best_params['y'], input_len=scaled_features.shape[1], sigma=best_params['sigma'], learning_rate=best_params['learning_rate'])
    som.random_weights_init(scaled_features)
    som.train_random(scaled_features, 100)
    win_map = som.win_map(scaled_features)
    clusters = np.zeros(len(scaled_features))
    for cluster, points in win_map.items():
        for point in points:
            idx = np.where((scaled_features == point).all(axis=1))[0][0]
            clusters[idx] = cluster[0] * som.get_weights().shape[1] + cluster[1]

    return clusters, 'som', 'SOM Clusters'

def kmeans_algorithm(scaled_features):
    def elbow_method(scaled_features, max_clusters=10):
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_features)
            wcss.append(kmeans.inertia_)
        return wcss

    def find_optimal_clusters(scaled_features, max_clusters=10):
        best_n_clusters = 2
        best_silhouette = -1
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_clusters = n_clusters
        return best_n_clusters

    wcss = elbow_method(scaled_features)
    optimal_clusters = find_optimal_clusters(scaled_features)

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    return clusters, 'kmeans', 'KMeans Clusters'

def dbscan_algorithm(scaled_features):
    def find_optimal_eps(scaled_features, min_samples=5):
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(scaled_features)
        distances, indices = neighbors_fit.kneighbors(scaled_features)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        optimal_eps = distances[np.argmax(np.diff(distances, 2))]
        return optimal_eps

    def find_optimal_min_samples(scaled_features, eps, max_samples=20):
        best_min_samples = 2
        best_silhouette = -1
        for min_samples in range(2, max_samples + 1):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(scaled_features)
            unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            if unique_clusters > 1:
                valid_indices = clusters != -1
                silhouette_avg = silhouette_score(scaled_features[valid_indices], clusters[valid_indices])
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_min_samples = min_samples
        return best_min_samples

    optimal_eps = find_optimal_eps(scaled_features)
    optimal_min_samples = find_optimal_min_samples(scaled_features, optimal_eps)

    dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
    clusters = dbscan.fit_predict(scaled_features)

    return clusters, 'dbscan', 'DBSCAN Clusters'

def agglomerative_algorithm(scaled_features):
    def find_optimal_clusters(scaled_features, max_clusters=20):
        best_n_clusters = 2
        best_silhouette = -1
        for n_clusters in range(2, max_clusters + 1):
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = agglomerative.fit_predict(scaled_features)
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_clusters = n_clusters
        return best_n_clusters

    optimal_clusters = find_optimal_clusters(scaled_features)

    agglomerative = AgglomerativeClustering(n_clusters=optimal_clusters)
    clusters = agglomerative.fit_predict(scaled_features)

    return clusters, 'agglomerative', 'Agglomerative Clusters'

def gmm_algorithm(scaled_features):
    def find_optimal_gmm_components(scaled_features, max_components=20):
        bics = []
        aics = []
        for n_components in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(scaled_features)
            bics.append(gmm.bic(scaled_features))
            aics.append(gmm.aic(scaled_features))
        optimal_n_components_bic = bics.index(min(bics)) + 1
        optimal_n_components_aic = aics.index(min(aics)) + 1
        return optimal_n_components_bic, optimal_n_components_aic

    optimal_n_components_bic, optimal_n_components_aic = find_optimal_gmm_components(scaled_features)
    optimal_n_components = optimal_n_components_bic  # You can choose based on BIC or AIC

    gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
    clusters = gmm.fit_predict(scaled_features)

    return clusters, 'gmm', 'GMM Clusters'

def hierarchical_algorithm(scaled_features):
    def apply_hierarchical_clustering(scaled_features, method='ward'):
        linkage_matrix = linkage(scaled_features, method=method)
        return linkage_matrix

    def determine_optimal_clusters(linkage_matrix, max_distance=10):
        distances = linkage_matrix[:, 2]
        distances_diff = np.diff(distances)
        optimal_distance = distances[np.argmax(distances_diff) + 1]
        if optimal_distance > max_distance:
            optimal_distance = max_distance
        return optimal_distance

    linkage_matrix = apply_hierarchical_clustering(scaled_features)
    optimal_distance = determine_optimal_clusters(linkage_matrix)
    clusters = fcluster(linkage_matrix, optimal_distance, criterion='distance')

    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'dendrogram.png'))
    plt.show()

    return clusters, 'hierarchical', 'Hierarchical Clusters'

def spectral_algorithm(scaled_features):
    def find_optimal_clusters(scaled_features, max_clusters=20):
        best_n_clusters = 2
        best_silhouette = -1
        for n_clusters in range(2, max_clusters + 1):
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
            cluster_labels = spectral.fit_predict(scaled_features)
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_clusters = n_clusters
        return best_n_clusters

    optimal_clusters = find_optimal_clusters(scaled_features)
    spectral = SpectralClustering(n_clusters=optimal_clusters, affinity='nearest_neighbors', random_state=42)
    clusters = spectral.fit_predict(scaled_features)

    return clusters, 'spectral', 'Spectral Clusters'

# Instantiate and run the clustering with multiple algorithms
algorithms = [
    som_algorithm, 
    kmeans_algorithm, 
    dbscan_algorithm, 
    agglomerative_algorithm, 
    gmm_algorithm, 
    hierarchical_algorithm, 
    spectral_algorithm
]

stock_clustering = StockClustering(algorithms)
stock_clustering.run(DATA_FILE_PATH)
