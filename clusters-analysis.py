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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Read created csv files from data-preparation file
def readCSV(filename):

    df = pd.read_csv(filename)

    return df

def histogramPlot(data):
    
    features = data.columns[1:-1]  # Exclude 'Consumer-ID' and 'Cluster'
    getClusters = data['Cluster'].unique() # Get list of possible clusters example: [1,2] if there are 2 clusters

    # Create a histogram for each feature in each cluster
    for cluster in getClusters:
        # Assign number of cluster
        clusterNumber = data[data['Cluster'] == cluster]
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'Cluster {cluster}', fontsize=14)

            # Set up the plotting area
        num_features = len(features)
        num_rows = (num_features // 7) + 1  # Adjust the number of rows based on features
    
        # Create histograms for each feature
        for i, feature in enumerate(features, 1):
            plt.subplot(num_rows, 7, i)  # Assuming there are at most 44 features to plot
            plt.hist(clusterNumber[feature], bins=10, edgecolor='k')
            plt.title(feature)
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'cluster_{cluster}_feature_distributions.png')
        plt.show()

def plot_single_feature(data, cluster):

    features = data.columns[1:-1]  # Exclude 'Consumer-ID' and 'Cluster'
    clusterNumber = data[data['Cluster'] == cluster]

    for feature in features:
        plt.figure(figsize=(8, 6))
        plt.hist(clusterNumber[feature], bins=10, edgecolor='k')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(True)

        # Save the plot to png
        filename = f'cluster_{cluster}_{feature}_distribution.png'
        plt.savefig(filename)
        print(f'Saved {filename}')
        
        plt.close()  # Close the plots

filename = '/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/SORTED.csv'
sortedDf = readCSV(filename)

filename = '/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/SORTEDkmeans.csv'
sortedDfKmeans = readCSV(filename)

# print(sortedDf)
# histogramPlot(sortedDf)
# plot_single_feature(sortedDf, 1)

# print(sortedDfKmeans)
# histogramPlot(sortedDfKmeans)

# plot_single_feature(0,'UI')

# Remove 'Consumer-ID' and 'Cluster' columns
data_prepared = sortedDf.drop(columns=['Consumer-ID', 'Cluster'])
data_normalized = (data_prepared - data_prepared.min()) / (data_prepared.max() - data_prepared.min())
print(data_normalized)
# Create the heat map
plt.figure(figsize=(12, 10))
sns.heatmap(data_normalized, cmap='coolwarm', cbar=True, yticklabels=sortedDf['Cluster'])
plt.title('Heatmap of Survey Responses')
plt.xlabel('Questions')
plt.ylabel('Respondents')

# save heatmap with higher dpi
plt.savefig('heatmap_survey_responses.png', dpi=300, bbox_inches='tight')
plt.show()