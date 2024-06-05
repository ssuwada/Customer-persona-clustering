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

# Function to compute wcss (Within-Cluster Sum of Square) 
def WCSS(segment, max_clusters):
    
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

def perform_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)

    labels = kmeans.labels_
    data['Cluster'] = labels

    data.to_csv(f'FinalCluster2.csv', index=False)


    return kmeans.labels_, kmeans.cluster_centers_

def last_kmeans_general(dataClusters):
    combined_features = dataClusters.values  # Convert DataFrame to numpy array
    print(combined_features)
    # Normalize the data
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined_features)

    # Perform K-means clustering
    k = 2  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_features)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Add cluster labels to the original DataFrame
    dataClusters['Cluster'] = cluster_labels
    dataClusters.to_csv(f'FinalCluster.csv', index=False)

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


# plot_elbow_df(dfCenters, 10)

# last_kmeans_general(dfCenters)

# columns_to_cluster = dfCenters.columns.tolist()
# data_for_clustering = dfCenters[columns_to_cluster]
# n_clusters = 2

# cluster_labels, cluster_centers = perform_kmeans_clustering(data_for_clustering, n_clusters)
# dfCenters['Cluster_Labels'] = cluster_labels
# print(dfCenters)

# for z in range(len(segemtns)):
#     segemtns[z]['ClustersLabels'] = dfCenters['Cluster_Labels']
#     print(segemtns[z]) 

###################
# # Transforming the DataFrame to long format
# long_df = pd.DataFrame()

# for i in range(1, 6):  # Adjust the range according to the number of segments
#     temp_df = pd.DataFrame({
#         'Centers_x': dfCenters[f'Segment{i}_Centers_x'],
#         'Centers_y': dfCenters[f'Segment{i}_Centers_y']
#     })
#     long_df = pd.concat([long_df, temp_df], ignore_index=True)

# print(long_df)

# df_final, centers_final = FinalKMeans(long_df,2)

# print(df_final)


# segment.to_csv(f'FinalCluster.csv', index=False)
