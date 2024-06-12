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
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import hypertools as hyp





# Read created csv files from data-preparation file
def read_csvs(filename, segments_numb):

    segments = []
    for i in range(1,segments_numb):
        df = pd.read_csv(filename.format(i))
        segments.append(df)

    return segments

# Function to compute wcss (Within-Cluster Sum of Square) 
def WCSS(segment, max_clusters):
    
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

# Elbow Metehod - defines number of clusters for DataFrame
def plot_elbow_df(df, max_clusters):
        
    wcss = WCSS(df, max_clusters)

    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method')
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

# Function to perform K-means clustering on a segment - apply PCA also 
def cluster_segment(segment, n_clusters, ColumnName, n_components):
    features = segment.drop(columns=['Consumer-ID'])
    reduced_features = apply_pca(features, n_components)
    # print(reduced_features)
    plot_elbow_df(reduced_features,10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    segment[ColumnName] = kmeans.fit_predict(reduced_features)
    # print("Interia of clusters: %d" %(kmeans.inertia_))
    plot_clusters(segment, kmeans.cluster_centers_, reduced_features)

    return segment

# Create plots of clusters based on reduced features created by PCA (transform clusters to 2D) - reduce dimensionality
def plot_clusters(segment, cluster_centers, reduced_features):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=segment['Cluster'], cmap='viridis', marker='o')
    if isinstance(cluster_centers, list):
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

# Create new dataFrame and store centers cooridantes there from each segments after initial clustering
def dfForCentersCluster(segments):
    data = {}
    for i in range(len(segments)):
        segment = segments[i]
        data[f'Segment{i+1}_Centers_x'] = segment['Cluster-segment-center-x']
        data[f'Segment{i+1}_Centers_y'] = segment['Cluster-segment-center-y']
    
    df = pd.DataFrame(data)

    # Plotting the points with different colors for each segment
    plt.figure(figsize=(10, 6))
    for i in range(len(segments)):
        plt.scatter(df[f'Segment{i+1}_Centers_x'], df[f'Segment{i+1}_Centers_y'], label=f'Segment {i+1}')
    
    plt.title('Centers of Segments')
    plt.xlabel('Centers_x')
    plt.ylabel('Centers_y')
    plt.legend()
    plt.show()

    return df

# Function to perform K-means clustering on a segment
def FinalKMeans(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['FinalCluster'] = kmeans.fit_predict(df)
    cluster_centers = kmeans.cluster_centers_

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Centers_x'], df['Centers_y'], c=df['FinalCluster'], s=50, cmap='viridis', label='Data Points')
    
    # Plot the cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Cluster Centers')
    plt.title(f'KMeans Clustering with {n_clusters} Clusters')
    plt.xlabel('Centers_x')
    plt.ylabel('Centers_y')
    plt.legend()
    plt.colorbar(scatter)
    plt.show()

    return df, cluster_centers

def last_kmeans_general(dataClusters, k):
    combined_features = dataClusters.values  # Convert DataFrame to numpy array
    n_components = ThresholdPcaRetain_singleSegment(dataClusters, 0.9)
    reduced_features = apply_pca(dataClusters, n_components)


    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(reduced_features)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Add cluster labels to the original DataFrame
    dataClusters['Cluster'] = cluster_labels
    dataClusters.to_csv(f'FinalCluster.csv', index=False)

    plot_clusters(dataClusters, kmeans.cluster_centers_, reduced_features)

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

## CALCULATE Silhouette Scoress for one segment (can be used for calculation for whole data set)
def silhouette_scoress(segment, max_clusters):
        
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

### PART 2 - Functions ###
def combine_segments(segments, key='Consumer-ID'):

    # Start with first DF - assign first to combined_df
    combined_df = segments[0]
    # Merge all DF using key='Consumer-ID'
    for df in segments[1:]:
        combined_df = combined_df.merge(df, on=key)
    
    return combined_df

def ThresholdPcaRetain_singleSegment(segment, thresholdExplainedVariance):
    
        # Standardize the segment data
    scaler = StandardScaler()
    segment_scaled = scaler.fit_transform(segment)
        
        # Perform PCA to obtain number of components to retain
    pca = PCA()
    princComponents = pca.fit_transform(segment_scaled)
        
        # Calculate cumulative explained variance
    cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
        
        # Determine the number of components to retain
    num_components = next(i for i, total_variance in enumerate(cumulative_explained_variance) if total_variance >= thresholdExplainedVariance) + 1
    print(len(segment))
    return num_components


## PART 3 DBSCAN

def dbscan_cluster(segment, n_components, epss, min_sampless):
    features = segment.drop(columns=['Consumer-ID'])
    reduced_features = apply_pca(features, n_components)
    db = DBSCAN(eps=epss, min_samples=min_sampless).fit(reduced_features)
    labels = db.labels_
    segment['Cluster'] = labels


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = reduced_features[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

    plt.title('DBSCAN Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

def plot_k_distance(data, k):
    # Calculate the k-nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Sort the distances (k-th nearest neighbors)
    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]

    # Plot the distances
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('k-Distance Graph')
    plt.xlabel('Data Points')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.show()


## PART 4 MEDOIDS (KMEDOIDS)
def medoids_cluster(segment, n_clusters, n_components):
    features = segment.drop(columns=['Consumer-ID'])
    reduced_features = apply_pca(features, n_components)

    # Compute Kmedoids clustering
    cobj = KMedoids(n_clusters).fit(reduced_features)
    labels = cobj.labels_
    segment['Cluster'] = cobj.fit_predict(reduced_features)


    print("Interia of clusters: %d" %(cobj.inertia_))
    plot_clusters(segment, cobj.cluster_centers_, reduced_features)


## PART 5  Hierarchical Cluster
def hierarchical_cluster(segment, n_clusters, n_components):
    features = segment.drop(columns=['Consumer-ID'])
    reduced_features = apply_pca(features, n_components)
    
    hierarchical = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
    segment['Cluster']  = hierarchical.fit_predict(reduced_features)
    labels = hierarchical.labels_

    linkage_data = linkage(reduced_features, method='ward', metric='euclidean')
    dendrogram(linkage_data)
    plt.show()
    print(labels)


    # plot_clusters(segment, 0, reduced_features)

    segment['Cluster'] = hierarchical.labels_
    hyp.plot(features, '.', group=labels, reduce='TSNE', ndims=3, legend=['Cluster 0', 'Cluster 1'])
    plt.show()

    return segment



## MAIN


### PART 1.1 ###
# Import segments into list

filename = '/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/S{}.csv'
segemtns = read_csvs(filename, 6) 


### PART 1.2 ###
# Define cluster sizes for each segments based on Elbow Method and Silhouette Scores
# plot_elbow_method(segemtns, 10)
# calculate_silhouette_scores(segemtns, 10)


### PART 1.3 ###
# Create Vector - reductionVector what stores number of 90% components that can retain based on Cumulative Explained Variance
reductionVector = ThresholdPcaRetain(segemtns, 0.9)


### PART 1.4 ###
## DEFINE VECTOR of Clusters for each segment for example: [3,3,2,3,3]
# It should be done based on elbow Method and Silhouette Score
clusterSizeVector = [2,2,2,2,2]
# Define name of new column in dataFrame for Cluster number
ColumnName='Cluster'


### PART 1.5 ### 
## Perform clustering based on Size of Clusers defined by Elbow method

### PART ADDITIONALES ####
# Create Clustering based on centers obtained from previous clustering!
# # Transforming the DataFrame to long format

# centers = segments_assign_centers(segemtns, clusterSizeVector, ColumnName, reductionVector)
# dfCenters = dfForCentersCluster(centers)
# last_kmeans_general(dfCenters,2)



## PART 2.1 ### - try clustering using all data - KMEANS

combinedDF = combine_segments(segemtns, 'Consumer-ID')
# print(combinedDF)
# plot_elbow_df(combinedDF,10)
reductionNumber = ThresholdPcaRetain_singleSegment(combinedDF, 0.9)
# print(reductionNumber)

ClusterinSingleSegment = cluster_segment(combinedDF, 2, ColumnName, reductionNumber)
# print(ClusterinSingleSegment)
# silhouette_scoress(combinedDF, 10)

### PART 3.1 - DBSCAN model for clusterin ###

# PLOT K-DISTANCE
features = combinedDF.drop(columns=['Consumer-ID'])
reduced_features = apply_pca(features, reductionNumber)
# plot_k_distance(reduced_features, k=5)

# dbscan_cluster(combinedDF, reductionNumber, 8.1, 5)

## DBSCAN IS NOT WORKING FOR THIS SPECYFIC SET OF DATA

### PART 4.1 - MEDOIDS CLUSTERING

# medoids_cluster(combinedDF, 2, reductionNumber)

### PART 5.1 - HIERARCHICAL

# HierdfCombined = hierarchical_cluster(combinedDF, 2, reductionNumber)

# SORT VALUES AND EXPORT FOR HIERRARCHICAL

# sorted_df = HierdfCombined.sort_values(by='Cluster', ascending=False)
# print(sorted_df)

# counts = sorted_df['Cluster'].value_counts()
# print(counts)

# sorted_df.to_csv(f'SORTED.csv', index=False)


# CREATE SORTED FILE FOR KMEANS CLUSTERING

sorted_df2 = ClusterinSingleSegment.sort_values(by='Cluster', ascending=False)
print(sorted_df2)

counts2 = sorted_df2['Cluster'].value_counts()
print(counts2)

sorted_df2.to_csv(f'SORTEDkmeans.csv', index=False)