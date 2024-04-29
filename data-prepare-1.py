#                       ---------------------
#                       ---------------------
#   ---------  Data preperation part - Customer Personas  ---------
#                  Master Thesis evalutaion project
#             
#                       ---------------------
#                          Sebastian Suwada
#                       ---------------------
#   Creation Date: 2024-04-29
#   Last Modified: 2024-04-29
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
import matplotlib.pyplot as plt


def prepareData(filename):
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
    print(df)
    # Get all columns from DataFrame
    columns = df.columns
    #print(columns)
    # Identify columns that end with "_Rating"
    rating_columns = [col for col in columns if col.endswith("_Rating")]
    #print(rating_columns)

    # Find the corresponding prefix columns (without "_Rating")
    prefixes = {col.rsplit("_", 1)[0] for col in rating_columns}
    
    # Keep columns that either are in the prefixes set and end with "_Rating"
    columns_to_keep = [col for col in columns if col in rating_columns or col.split("_Rating")[0] in prefixes]
    
    # Create a new DataFrame with the filtered columns
    filtered_df = df[columns_to_keep]
    print(filtered_df[5:10])

    return 

#   ---------  Main  ---------

prepareData('/Users/sebastiansuwada/Desktop/HTB/McsThesis/Code-Thesis/Customer-persona-clustering/data1.xlsx')

#biore tez beta-alanine

