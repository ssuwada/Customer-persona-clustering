#                       ---------------------
#                       ---------------------
#   ---------  Data preperation part - Customer Personas  ---------
#                  Master Thesis evalutaion project
#             
#                       ---------------------
#                          Sebastian Suwada
#                       ---------------------
#   Creation Date: 2024-04-29
#   Last Modified: 2024-05-13
#
#   Description:
#       This part of the project evaluates the data preperation part.
#       During collecting data using cognitoforms.com for project 
#       "Survey of consumers' travel and mobile app preferences and behavior",
#       aim was to collect information from travel active people about
#       their trip prepraing behaviour. Based on that in this file (what is first part I would
#       like to prepare data for later analysis). Basic transofrmation from "human-names" to 
#       more "computer-friendly-names".


#   ---------  Import libraries part  ---------

import numpy as np
import pandas as pd
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def initialPrepareData(filename):
    """
    Function: Data preparation
    Created: 2024-04-29
    Last Modified: 2024-04-29

    Description:

    Keywords:
        data cleaning, data preparing, cleaning, database transform

    Parameters:
        data (DataFrame): The dataset on which the clustering will be perfromed

    Returns:
        results (DataFrame): As results gives back dataFrame of new prepared data
    """

    df = pd.read_excel(filename)

    # Filter to keep only columns that end with '_Rating'
    rating_columns = [col for col in df.columns if col.endswith('_Rating')]

    # Create variable that will speicy which columns would be removed
    columns_to_remove = {col.split('_Rating')[0] for col in rating_columns if col.split('_Rating')[0] in df.columns}

    # Drop columns to remove that are duplicated
    df = df.drop(columns=columns_to_remove)

    # Drop last 4 columns that inlcude information about submission, date and hour
    df = df.drop(columns=df.columns[-4:])

    # Save it to csv for inspection of code work
    df.to_csv("filtered_output.csv", index=False)

    return df

def getSegements(initdf):

    """
    Function: Substracting segments from general dataFrame
    Created: 2024-05-13
    Last Modified: 2024-05-13

    Description:

    Keywords:
        data cleaning, data preparing, cleaning, database transform

    Parameters:
        data (DataFrame): Initial dataFrame cleanned by initialPrepareData()

    Returns:
        results (DataFrame): As results gives back dataFrame's of new prepared data for each segment
        separetly 
    """

    #    First segment S1
    ## Mobile applciations preferences ##
    # Segement "APPS PREF" include columns from 0 - 7 (6 varaibles and first ID)
    # Respond to the sentences based on your own experience: 
    # (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree and 5 = strongly agree).

    S1 = initdf.iloc[:, 0:7]
    S1.columns = ['Consumer-ID', 'UI', 'UX', 'Download-importance', 'Dark-Light', 'Logo-importance', 'Notifications']
    # print(S1)

    #    Second segment S2
    ## Culture experience during travel ##
    # Segement "Culture experience" include columns from 9 - 13 (4 varaibles and first ID)
    # Respond to the sentences based on your own experience: 
    # (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree and 5 = strongly agree).

    S2 = pd.concat([initdf.iloc[:, [0]], initdf.iloc[:, 9:13]], axis=1)
    S2.columns = ['Consumer-ID', 'Local-community', 'Local-events', 'Local-culture-experience', 'Historical-aspects']
    # print(S2)

    #    Third segment S3
    ## Social media aspects ##
    # Segement "Social Media" include columns from 13 - 32 (19 varaibles and first ID)
    # Respond to the sentences based on your own experience: 
    # (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree and 5 = strongly agree).

    S3 = pd.concat([initdf.iloc[:, [0]], initdf.iloc[:, 14:32]], axis=1)
    S3.columns = ['Consumer-ID', 'Every-day-Posting', 'Travel-for-account-promotion', 'Travel-influ-posting', 'Posting-during-travel', 'My-posting-influence-travel', 'I-know-where-to-travel', 'Search-for-travel-media', 'Media-are-suffic', 'UGC-travel-suff', 'Media-important-for-travel-search', 'I-would-use-travel-app', 'I-use-travel-portals', 'I-traveled-to-place-from-social-media', 'Adverts-on-trip', 'I-use-travel-apps', 'I-will-share-photos-during-trip', 'I-can-create-travel-plan', 'There-is-need-of-travel-service']
    # print(S3)

    #    Fourth segment S4
    ## Social media aspects ##
    # Segement "Social Media" include columns from 32 - 40 (8 varaibles and first ID)
    # Respond to the sentences based on your own experience: 
    # (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree and 5 = strongly agree).

    S4 = pd.concat([initdf.iloc[:, [0]], initdf.iloc[:, 32:40]], axis=1)
    S4.columns = ['Consumer-ID', 'Stick-budget', 'Sea-view', 'Budget-for-apartament', 'Budget-for-culture', 'Only-Restauratns', 'Spontanous-buying', 'BudgetTrip2024', 'Most-expensive-trip']
    # print(S4)

    #    5th segment S5
    ## Social media aspects ##
    # Segement "Social Media" include columns from 32 - 40 (8 varaibles and first ID)
    # Respond to the sentences based on your own experience: 
    # (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree and 5 = strongly agree).

    S5 = pd.concat([initdf.iloc[:, [0]], initdf.iloc[:, 40:46]], axis=1)
    S5.columns = ['Consumer-ID', 'Abroad-feeling', 'Routine-forget', 'Same-experience-all-the-time', 'Same-experience-all-the-time-v2', 'Safety-during-trip', 'Own-experience']
    # print(S5)

    return S1, S2, S3, S4, S5

def getDemographicalData(initdf):

    """
    Function: Substracting demographic informations from cleanned general dataFrame
    Created: 2024-05-13
    Last Modified: 2024-05-13

    Description:

    Keywords:
        data cleaning, data preparing, cleaning, database transform

    Parameters:
        data (DataFrame): Initial dataFrame cleanned by initialPrepareData()

    Returns:
        results (DataFrame): As results gives back dataFrame's of new prepared data with demographic 
        informations
    """

    demographicInformation = pd.concat([initdf.iloc[:, [0]], initdf.iloc[:, 7:9], initdf.iloc[:, 46:53]], axis=1)
    demographicInformation.columns = ['Consumer-ID', 'OS-Mobile', 'Mobile-time', 'Age', 'Gender', 'Education-level', 'Maritial-status', 'Kids', 'Income', 'Work-status']

    return demographicInformation

def saveSegmentstoCSV(segments, segments_names):
    """
    Function: Save dataFrames into csv
    Created: 2024-05-13
    Last Modified: 2024-05-13

    Description

    Keywords:
        data cleaning, data preparing, cleaning, database transform

    Parameters:
        data (DataFrame): Initial dataFrame cleanned by initialPrepareData()

    Returns:
        results (DataFrame): As results gives back dataFrame's of new prepared data with demographic 
        informations
    """
    # Save each segment to a different CSV file
    for segment, name in zip(segments, segments_names):
        segment.to_csv(f'{name}.csv', index=False)

#   ---------  Main  ---------

# Get data and make initial preparation
filename = '/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/data1.xlsx'
df = initialPrepareData(filename)

# Devide into segments(1-5) - create different dataFrames for each segment
segments = getSegements(df)
segment_names = ['S1', 'S2', 'S3', 'S4', 'S5']
saveSegmentstoCSV(segments, segment_names)

# Substract from main data, demographic informations
D1 = getDemographicalData(df)


# To other file - Create standarization of segments and perform PCA on each segment
scaler = StandardScaler()
segment_scaled = scaler.fit_transform(segments[0])
# print(segment_scaled)

pca = PCA()
principal_components = pca.fit_transform(segment_scaled)
explained_variance = pca.explained_variance_ratio_


