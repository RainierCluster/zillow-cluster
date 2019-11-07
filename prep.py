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

def pretty_cols(df):
    # better understood column names
    df = df.rename(columns={"calculatedfinishedsquarefeet":"house_area", "fips":"countyid", "structuretaxvaluedollarcnt":"house_value", "landtaxvaluedollarcnt":"land_value", "taxvaluedollarcnt":"whole_value", "lotsizesquarefeet":"whole_area"})

    # reorder columns based on relevancy to others
    df = df[["countyid","latitude","longitude","yearbuilt","bathroomcnt","bedroomcnt","house_area", "house_value","land_value","whole_area","whole_value","taxamount","logerror","transactiondate"]]
    return df

def cal_taxrate(df):
    df["taxrate"] = df.taxamount / df.whole_value
    df.drop(columns=["taxamount"], inplace=True)
    return df

def impute_lotsize_nulls(train, test):
    """
    Calculate a proportion from lot size and tax value. Disregard outliers, and calculate a mean proportion. Mulitple that mean proportion with the tax value to impute lot sizes for null values. 
    """
    # create subset dataframe with lot size and tax value
    df_subset = train [["whole_area", "whole_value"]]
    # create a new column that takes the proportion of lot size to tax value
    df_subset["proportionwhole"] = df_subset.whole_area.dropna() / df_subset.whole_value
    # get the average mean of proportions that are less than 1 and add it to a column
    mean_proportion = df_subset [df_subset.proportionwhole < 1].proportionwhole.mean()

    train["mean"] = mean_proportion
    test["mean"] = mean_proportion
     
    # fill all the nulls in lot saize with the calculated mean
    train.whole_area.fillna(train.whole_value * train["mean"], inplace=True)
    test.whole_area.fillna(test.whole_value * test["mean"], inplace=True)

    # clean
    train.whole_area = train.whole_area.round()
    test.whole_area = test.whole_area.round()

    train.drop(columns="mean", inplace=True)
    test.drop(columns="mean", inplace=True)
    return train, test

def cal_land_area(df):
    df["land_area"] = df.whole_area - df.house_area
    return df