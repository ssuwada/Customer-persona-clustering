#                       ---------------------
#                       ---------------------
#   ---------  Data clustering part - Customer Personas  ---------
#                  Master Thesis evalutaion project

#   ---------  Import libraries part  ---------

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import hypertools as hyp
from yellowbrick.cluster import KElbowVisualizer



###### FUNCTIONS ######

def readCSVS(filename, segments_numb):
# Read created csv files from data-preparation file
    segments = []
    for i in range(1,segments_numb):
        df = pd.read_csv(filename.format(i))
        segments.append(df)

    return segments

def ApplyPCA(features, n_components):
# Function to perform PCA and reduce dimensions based on previous threshold (90%)
    # Standardize the segment data
    scaler = StandardScaler()
    segment_scaled = scaler.fit_transform(features)

    # Perfomr PCA on scaled data for each segment
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(segment_scaled)

    return reduced_features

def WCSS(segment, max_clusters):
# Function to compute wcss (Within-Cluster Sum of Square) 
    if isinstance(segment, pd.DataFrame):
        if 'Consumer-ID' in segment.columns:
            segment = segment.drop(columns=['Consumer-ID'])

    wcss = []

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(segment)
        wcss.append(kmeans.inertia_)
        # Inertia = Σ(distance(point, centroid)^2)
    return wcss

def ElbowMethod(df, max_clusters):
# Elbow Metehod - defines number of clusters for DataFrame
    wcss = WCSS(df, max_clusters)

    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.legend()
    plt.show()

    # model = AgglomerativeClustering(metric='euclidean', linkage='ward')
    # model = KMeans()
    # visualizer = KElbowVisualizer(model, k=(2, 30), metric='calinski_harabasz', timings=True)
    # visualizer.fit(df)
    # visualizer.show()


def SilhouetteScoress(segment, max_clusters):
## CALCULATE Silhouette Scoress for one segment (can be used for calculation for whole data set)
 
    silhouette_scores = []

    for Numberclusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=Numberclusters, random_state=0)
        cluster_labels = kmeans.fit_predict(segment)
        silhouette_avg = silhouette_score(segment, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(silhouette_scores)

    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for different number of clusters')
    plt.grid(True)
    plt.show()

def CombineSegments(segments, key='Consumer-ID'):
    # Start with first DF - assign first to combined_df
    combined_df = segments[0]
    # Merge all DF using key='Consumer-ID'
    for df in segments[1:]:
        combined_df = combined_df.merge(df, on=key)
    
    return combined_df

def ThresholdPCARetain(segment, thresholdExplainedVariance):
    # Standardize data
    scaler = StandardScaler()
    segment_scaled = scaler.fit_transform(segment)
    
    # Perform PCA to get number of components to retain
    pca = PCA()
    princComponents = pca.fit_transform(segment_scaled)
        
    # Calculate cumulative explained variance
    cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
        
    # Determine the number of components to retain
    num_components = next(i for i, total_variance in enumerate(cumulative_explained_variance) if total_variance >= thresholdExplainedVariance) + 1
    print(len(segment))

    # cumulative explained variance with components
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.axhline(y=thresholdExplainedVariance, color='r', linestyle='-', label=f'({thresholdExplainedVariance * 100:.0f}%) Threshold')
    plt.axhline(y=thresholdExplainedVariance-0.1, color='b', linestyle='-', label=f'({(thresholdExplainedVariance-0.1) * 100:.0f}%) Threshold')
    plt.axvline(x=num_components, color='g', linestyle='--', label=f'{num_components} Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Components & CEV')
    plt.grid(True)
    plt.legend()
    plt.show()

    return num_components


def PlotClusters(segment, cluster_centers, reduced_features):
# Create plots of clusters based on reduced features created by PCA (transform clusters to 2D) - reduce dimensionality
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=segment['Cluster'], cmap='viridis', marker='o')
    if isinstance(cluster_centers, list):
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')  # Plotting cluster centers
    plt.title('Clusters (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

## FUNCTION FOR CLUSTERING - KMEANS ##

def ClusterKMEANS(segment, n_clusters, ColumnName, n_components):
# Function to perform K-means clustering on a segment - apply PCA also 
    features = segment.drop(columns=['Consumer-ID'])
    reduced_features = ApplyPCA(features, n_components)
    ElbowMethod(reduced_features,10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    segment[ColumnName] = kmeans.fit_predict(reduced_features)
    # PlotClusters(segment, kmeans.cluster_centers_, reduced_features)

    return segment

## FUNCTION FOR HIERARCHICAL CLUSTERING ##

def hierarchical_cluster(segment, n_clusters, n_components):
    features = segment.drop(columns=['Consumer-ID'])
    reduced_features = ApplyPCA(features, n_components)
    
    hierarchical = AgglomerativeClustering(n_clusters, metric='euclidean', linkage='ward')
    segment['Cluster']  = hierarchical.fit_predict(reduced_features)
    labels = hierarchical.labels_

    linkage_data = linkage(reduced_features, method='ward', metric='euclidean')
    dendrogram(linkage_data)
    plt.show()

    segment['Cluster'] = hierarchical.labels_
    hyp.plot(features, '.', group=labels, reduce='TSNE', ndims=3, legend=['Cluster 0', 'Cluster 1'])
    plt.show()

    return segment


###### MAIN ######


#   PART 1 - Import segments into list
filename = '/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/S{}.csv'
segemtns = readCSVS(filename, 6) 

#   PART 2  - clustering using all segments in variable combineDF
combinedDF = CombineSegments(segemtns, 'Consumer-ID')
ElbowMethod(combinedDF,10)
reductionNumber = ThresholdPCARetain(combinedDF, 0.9)
print(reductionNumber)
SilhouetteScoress(combinedDF, 10)
reducedFS = ApplyPCA(combinedDF, reductionNumber)
SilhouetteScoress(reducedFS, 10)

#   PART 3 - perform clustering KMEANS
ColumnName='Cluster'
ClusteringCombinedKMEANS = ClusterKMEANS(combinedDF, 2, ColumnName, reductionNumber)
print(ClusteringCombinedKMEANS)

### PART 4 - HIERARCHICAL CLUSTERING

HierdfCombined = hierarchical_cluster(combinedDF, 2, reductionNumber)


##### EXPORT TO FILES ####

# SORT VALUES AND EXPORT FOR HIERRARCHICAL
sorted_df = HierdfCombined.sort_values(by='Cluster', ascending=False)
print(sorted_df)
counts = sorted_df['Cluster'].value_counts()
print(counts)
sorted_df.to_csv(f'SORTED.csv', index=False)


# CREATE SORTED FILE FOR KMEANS CLUSTERING
sorted_df2 = ClusteringCombinedKMEANS.sort_values(by='Cluster', ascending=False)
print(sorted_df2)
counts2 = sorted_df2['Cluster'].value_counts()
print(counts2)
sorted_df2.to_csv(f'SORTEDkmeans.csv', index=False)