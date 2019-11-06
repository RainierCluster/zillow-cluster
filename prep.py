import pandas as pd 
import numpy as numpy

import warnings
warnings.filterwarnings("ignore")

def handle_missing_values(df, prop_required_column, prop_required_row):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def clean_columns(df):
    """
    Removing columns that were determined not to provide value
    """
    df.drop(columns=["calculatedbathnbr", "fullbathcnt", "finishedsquarefeet12", "regionidcounty", "propertycountylandusecode", "rawcensustractandblock", "censustractandblock", "regionidzip", "regionidcity", "assessmentyear", "propertylandusedesc", "parcelid","roomcnt"], inplace=True)
    return df

def drop_minimal_nulls(df):
    """
    Drop rows that have that minimal nulls  
    """
    df.dropna(axis=0,subset=["calculatedfinishedsquarefeet","structuretaxvaluedollarcnt", "taxvaluedollarcnt","landtaxvaluedollarcnt","taxamount", "yearbuilt"], inplace=True)
    return df

def impute_lotsize_nulls(df):
    """
    Calculate a proportion from lot size and tax value. Disregard outliers, and calculate a mean proportion. Mulitple that mean proportion with the tax value to impute lot sizes for null values. 
    """
    # create subset dataframe with lot size and tax value
    df_subset = df [["lotsizesquarefeet", "taxvaluedollarcnt"]]
    # create a new column that takes the proportion of lot size to tax value
    df_subset["proportionlotandvalue"] = df_subset.lotsizesquarefeet.dropna() / df_subset.taxvaluedollarcnt
    # get the average mean of proportions that are less than 1 and add it to a column
    mean_proportion = df_subset [df_subset.proportionlotandvalue < 1].proportionlotandvalue.mean()
    df_subset["mean"] = mean_proportion
    # fill all the nulls in lot saize with the calculated mean
    df_subset.lotsizesquarefeet.fillna(df_subset.taxvaluedollarcnt * df_subset["mean"], inplace=True)
    # replace old lot size with null-free lot size to the original dataframe 
    df["lotsizesquarefeet"] = df_subset.lotsizesquarefeet.round(0)
    return df

def pretty_cols(df):
    # better understood column names
    df = df.rename(columns={"calculatedfinishedsquarefeet":"house_area", "fips":"countyid", "structuretaxvaluedollarcnt":"house_value", "landtaxvaluedollarcnt":"land_value", "taxvaluedollarcnt":"whole_value", "lotsizesquarefeet":"whole_area"})
    # reorder columns based on relevancy to others
    df = df[["countyid","latitude","longitude","yearbuilt","bathroomcnt","bedroomcnt","house_area","house_value","whole_area","whole_value","land_value","taxamount","logerror","transactiondate"]]
    return df