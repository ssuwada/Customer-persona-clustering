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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score



# Read created csv files from data-preparation file
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

# Elbow Metehod - defines number of clusters
def plot_elbow_method(segments, max_clusters):

    for j in range(len(segments)):
        wcss = WCSS(segments[j], max_clusters)

        plt.plot(range(1, max_clusters + 1), wcss, marker='o')
        plt.title('Elbow Method for segment '+str(j+1))
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.legend()
        plt.show()

# Create a function to perform PCA and retain the necessary components
def ThresholdPcaRetain(segment, thresholdExplainedVariance):
    
    num_components_to_retain = []

    for i in range(len(segemtns)):
        # Standardize the segment data
        scaler = StandardScaler()
        segment_scaled = scaler.fit_transform(segment[i])
        
        # Perform PCA to obtain number of components to retain
        pca = PCA()
        princComponents = pca.fit_transform(segment_scaled)
        
        # Calculate cumulative explained variance
        cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
        
        # Determine the number of components to retain
        num_components = next(i for i, total_variance in enumerate(cumulative_explained_variance) if total_variance >= thresholdExplainedVariance) + 1
        num_components_to_retain.append(num_components)

    return num_components_to_retain

# Function to perform K-means clustering on a segment
def cluster_segment(segment, n_clusters, ColumnName, n_components):
    features = segment.drop(columns=['Consumer-ID'])
    reduced_features = apply_pca(features, n_components)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    segment[ColumnName] = kmeans.fit_predict(reduced_features)
    return segment, kmeans.cluster_centers_, reduced_features

# Create plots of clusters based on reduced features created by PCA (transform clusters to 2D) - reduce dimensionality
def plot_clusters(segment, cluster_centers, reduced_features):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=segment['Cluster'], cmap='viridis', marker='o')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')  # Plotting cluster centers
    plt.title('Clusters (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

# Function to perform PCA and reduce dimensions based on previous threshold (90%)
def apply_pca(features, n_components):
    # Standardize the segment data
    scaler = StandardScaler()
    segment_scaled = scaler.fit_transform(features)

    # Perfomr PCA on scaled data for each segment
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(segment_scaled)

    return reduced_features

# Get clusters using KMeans
def segments_assign_centers(segemtns, clusterSizeVector, ColumnName, reductionVector):

    for i in range(len(segemtns)):
        Cluster_segment, centers, reduced_features = cluster_segment(segemtns[i], clusterSizeVector[i], ColumnName, reductionVector[i])
        # plot_clusters(Cluster_segment, centers, reduced_features)
        segemtns[i]['Cluster'] = Cluster_segment['Cluster']
        segemtns[i]['Cluster-segment-center-x'] = Cluster_segment['Cluster']
        segemtns[i]['Cluster-segment-center-y'] = Cluster_segment['Cluster']

        for j in range(len(segemtns[i]['Cluster'])):
            cluster_index = segemtns[i]['Cluster'][j]
            segemtns[i].loc[j, 'Cluster-segment-center-x'] = centers[cluster_index][0]
            segemtns[i].loc[j, 'Cluster-segment-center-y'] = centers[cluster_index][1]

        print(segemtns[i].head())

    return segemtns

## Centers of cluster segments to one DF
def listForCentersCluster(segments):
    df = pd.DataFrame()
    lista = []
    for i in range(len(segments)):
        segment = segments[i]
        temp = []
        for j in range(len(segment)):
            temp = [segment['Cluster-segment-center-x'][j], segment['Cluster-segment-center-y'][j]]
            print(temp)
            # df['SegmentCenter-'+str(i)] = temp
            lista.append(temp)
        # df['SegmentCenter-Y'+str(i)] = segment['Cluster-segment-center-y']
    
    return lista

# Function to perform K-means clustering on a segment
def FinalKMeans(lista, n_clusters):
    segment = pd.DataFrame()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    segment['FinalCluster'] = kmeans.fit_predict(lista)
    print(segment)
    return segment, kmeans.cluster_centers_

# Function to plot the elbow curve
def plot_elbowv2(data, max_clusters):
    distortions = []
    K = range(1, max_clusters+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal Number of Clusters')
    plt.show()

def calculate_silhouette_scores(segments, max_clusters=10):
    
    for i in range(len(segemtns)):

        silhouette_scores = []

        for Numberclusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=Numberclusters, random_state=0)
            cluster_labels = kmeans.fit_predict(segments[i])
            silhouette_avg = silhouette_score(segments[i], cluster_labels)
            silhouette_scores.append(silhouette_avg)
        print(silhouette_scores)

        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores for different number of clusters')
        plt.grid(True)
        plt.show()

    return silhouette_scores

## MAIN

filename = '/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/S{}.csv'
segemtns = read_csvs(filename, 6) 


# Define cluster sizes for each segments based on Elbow Method and Silhouette Scores
# plot_elbow_method(segemtns, 10)
# calculate_silhouette_scores(segemtns, 10)

# Create Vector - reductionVector what stores number of 90% components that can retain based on Cumulative Explained Variance
reductionVector = ThresholdPcaRetain(segemtns, 0.8)

## DEFINE VECTOR of Clusters for each segment for example: [3,3,2,3,3]
# It should be done based on elbow Method
clusterSizeVector = [2,2,2,2,2]

# Define name of new column in dataFrame for Cluster number
ColumnName='Cluster'

## Perform clustering based on Size of Clusers defined by Elbow method
# reductionVector defined by PCA and Cumulative Explained Variance results
segments = segments_assign_centers(segemtns, clusterSizeVector, ColumnName, reductionVector)



# Tu ponizej jest do poprawy, musze uwzglednic zeby bylo 205 r0w i 5 column z srodkami tych centrow
lista = listForCentersCluster(segments)
plot_elbowv2(lista,10)
segment, cluster_centers = FinalKMeans(lista, 2)

# Separate the points into X and Y coordinates
x_coords = [point[0] for point in lista]
y_coords = [point[1] for point in lista]

plt.scatter(x_coords, y_coords, c=segment['FinalCluster'], cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='x')  # Plot cluster centers
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.title('Scatter Plot with Cluster Centers')
plt.grid(True)
plt.show()


segment.to_csv(f'FinalCluster.csv', index=False)
