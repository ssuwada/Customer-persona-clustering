#                       ---------------------
#                       ---------------------
#   ---------  Data clustering part - Customer Personas  ---------
#                  Master Thesis evalutaion project
#             
#                       ---------------------
#                          Sebastian Suwada
#                       ---------------------
#   Creation Date: 2024-05-19
#   Last Modified: 2024-05-19
#
#   Description:



#   ---------  Import libraries part  ---------

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_csvs(filename, segments_numb):

    segments = []
    for i in range(1,segments_numb):
        df = pd.read_csv(filename.format(i))
        segments.append(df)

    return segments

# Function to compute wcss (Within-Cluster Sum of Square) 
def WCSS(segment, max_clusters):
    
    features = segment.drop(columns=['Consumer-ID'])

    wcss = []

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
        # Inertia = Σ(distance(point, centroid)^2)
    return wcss

def plot_elbow_method(segments, max_clusters):

    for j in range(len(segments)):
        wcss = WCSS(segments[j], max_clusters)

        plt.plot(range(1, max_clusters + 1), wcss, marker='o')
        plt.title('Elbow Method for segment '+str(j+1))
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.legend()
        plt.show()

# Function to perform K-means clustering on a segment
def cluster_segment(segment, n_clusters):
    features = segment.drop(columns=['Consumer-ID'])
    reduced_features = apply_pca(features, n_components=2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    segment['Cluster'] = kmeans.fit_predict(reduced_features)
    return segment, kmeans.cluster_centers_, reduced_features


def plot_clusters(segment, cluster_centers, reduced_features):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=segment['Cluster'], cmap='viridis', marker='o')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')  # Plotting cluster centers
    plt.title('Clusters (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

# Function to perform PCA and reduce dimensions
def apply_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features

filename = '/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/S{}.csv'
segemtns = read_csvs(filename, 6) 


# Define cluster sizes for each segments based on Elbow Method
# plot_elbow_method(segemtns, 10)

Cluster_segment1, centers, reduced_features = cluster_segment(segemtns[2], 3)

plot_clusters(Cluster_segment1, centers, reduced_features)

Cluster_segment1, centers, reduced_features = cluster_segment(segemtns[2], 2)

plot_clusters(Cluster_segment1, centers, reduced_features)