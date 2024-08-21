#                       ---------------------
#                       ---------------------
#   ---------  Data clustering part - Customer Personas  ---------
#                  Master Thesis evalutaion project
#             
#                       ---------------------
#                          Sebastian Suwada
#                       ---------------------
#   Creation Date: 2024-07-21
#   Last Modified: 2024-05-19
#
#   Description:



#   ---------  Import libraries part  ---------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def readCSV(filename):

    df = pd.read_csv(filename)

    return df

def analyze_cluster(cluster_number, D1, hierr):

    merged_df = pd.merge(D1, hierr[['Consumer-ID', 'Cluster']], on='Consumer-ID', how='left')

    cluster_df = merged_df[merged_df['Cluster'] == cluster_number]
    print(f"\nAnalysis for Cluster {cluster_number}:\n")

    # Create a list to store clusters info
    data = []
    
    gender_count = cluster_df['Gender'].value_counts()
    print("Gender:")
    print(gender_count)
    data.append(gender_count)

    osmobile_count = cluster_df['OS-Mobile'].value_counts()
    print("OS-Mobile distribution:")
    print(osmobile_count)
    data.append(osmobile_count)

    mobileTime_count = cluster_df['Mobile-time'].value_counts()
    print(mobileTime_count)
    data.append(mobileTime_count)

    age_count = cluster_df['Age'].value_counts()
    print(age_count)
    data.append(age_count)

    education_count = cluster_df['Education-level'].value_counts()
    print(education_count)
    data.append(education_count)

    martial_count = cluster_df['Maritial-status'].value_counts()
    print(martial_count)
    data.append(martial_count)

    income_count = cluster_df['Income'].value_counts()
    print(income_count)
    data.append(income_count)

    workStatus_count = cluster_df['Work-status'].value_counts()
    print(workStatus_count)
    data.append(workStatus_count)

    # Make it dataframe
    dataClusterDF = pd.concat(data, axis=1)

    # Save the DataFrame to a CSV file
    output_file = f'cluster_{cluster_number}_analysis.csv'
    dataClusterDF.to_csv(output_file)


# Function to append data to summary_dict
def append_summary(feature, value_counts, summary_dict, total_respondents):
    for category, count in value_counts.items():
        summary_dict['Feature'].append(feature)
        summary_dict['Category'].append(category)
        summary_dict['Number of Respondents'].append(count)
        summary_dict['Percentage of the Sample'].append(f"{(count / total_respondents) * 100:.2f}%")
    return summary_dict

def analyze_demographics(df):
    # Analyze the data
    os_mobile_counts = df['OS-Mobile'].value_counts()
    mobile_time_counts = df['Mobile-time'].value_counts()
    age_counts = df['Age'].value_counts()
    education_level_counts = df['Education-level'].value_counts()
    maritial_status_counts = df['Maritial-status'].value_counts()
    income_counts = df['Income'].value_counts()
    work_status_counts = df['Work-status'].value_counts()

    summary_dict = {
    'Feature': [],
    'Category': [],
    'Number of Respondents': [],
    'Percentage of the Sample': []
    }

    summary_dict = append_summary('OS-Mobile', os_mobile_counts, summary_dict, total_respondents=len(df))
    summary_dict = append_summary('Mobile-time', mobile_time_counts, summary_dict, total_respondents=len(df))
    summary_dict = append_summary('Age', age_counts, summary_dict, total_respondents=len(df))
    summary_dict = append_summary('Education-level', education_level_counts, summary_dict, total_respondents=len(df))
    summary_dict = append_summary('Maritial-status', maritial_status_counts, summary_dict, total_respondents=len(df))
    summary_dict = append_summary('Income', income_counts, summary_dict, total_respondents=len(df))
    summary_dict = append_summary('Work-status', work_status_counts, summary_dict, total_respondents=len(df))


    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv('summary_table_D1.csv', index=False)

filename = '/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/D1.csv'
D1 = readCSV(filename) #upload demographics

filename = '/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/SORTED.csv'
hierr = readCSV(filename) #upload clustered data


analyze_cluster(1, D1, hierr)
analyze_demographics(D1)


