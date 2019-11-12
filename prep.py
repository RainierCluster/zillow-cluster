import pandas as pd 
import numpy as numpy

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

def handle_missing_values(df, prop_required_column, prop_required_row):
    """
    Remove columns, then rows, that do not meet the set populated percentages.
    """
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def clean_columns(df):
    """
    Remove columns that were previously determined not to provide value.
    """
    df.drop(columns=["calculatedbathnbr", "fullbathcnt", "finishedsquarefeet12", "regionidcounty", "propertycountylandusecode", "rawcensustractandblock", "censustractandblock", "regionidzip", "regionidcity", "assessmentyear", "propertylandusedesc", "parcelid","roomcnt"], inplace=True)
    return df

def drop_minimal_nulls(df):
    """
    Drop rows that were previously determined to have a insignificant number of nulls.  
    """
    df.dropna(axis=0,subset=["calculatedfinishedsquarefeet","structuretaxvaluedollarcnt", "taxvaluedollarcnt","landtaxvaluedollarcnt","taxamount", "yearbuilt"], inplace=True)
    return df

def pretty_cols(df):
    """
    Rename columns to easier to understand names. Then reorder columns based on relevancy to others.
    """
    # rename
    df = df.rename(columns={"calculatedfinishedsquarefeet":"house_area", "fips":"countyid", "structuretaxvaluedollarcnt":"house_value", "landtaxvaluedollarcnt":"land_value", "taxvaluedollarcnt":"whole_value", "lotsizesquarefeet":"whole_area"})

    # reorder
    df = df[["countyid","latitude","longitude","yearbuilt","bathroomcnt","bedroomcnt","house_area", "whole_area", "house_value","land_value","whole_value","taxamount","logerror","transactiondate"]]
    return df

def cal_taxrate(df):
    """
    Create a new taxrate column and drop the taxamount column, as it no longer adds value.
    """
    df["taxrate"] = df.taxamount / df.whole_value
    df.drop(columns=["taxamount"], inplace=True)
    return df

def impute_lotsize_nulls(train, test):
    """
    Calculate a proportion from whole_area and whole_value. Disregard outliers and calculate a mean proportion. Mulitple that mean proportion with the whole_value to impute whole_area for null values. 
    """
    # create subset dataframe
    df_subset = train [["whole_area", "whole_value"]]
    # create a new column that takes the proportion
    df_subset["proportionwhole"] = df_subset.whole_area.dropna() / df_subset.whole_value
    # get the average mean of proportions that are less than 1 and add it to a column
    mean_proportion = df_subset [df_subset.proportionwhole < 1].proportionwhole.mean()

    train["mean"] = mean_proportion
    test["mean"] = mean_proportion
     
    # fill all the nulls in whole_area with the calculated mean
    train.whole_area.fillna(train.whole_value * train["mean"], inplace=True)
    test.whole_area.fillna(test.whole_value * test["mean"], inplace=True)

    # clean
    train.whole_area = train.whole_area.round()
    test.whole_area = test.whole_area.round()
    train.drop(columns="mean", inplace=True)
    test.drop(columns="mean", inplace=True)
    return train, test

def cal_land_area(df):
    """
    Derive a new column, land_area, from whole_area and house_area. Reorder columns. 
    """
    df["land_area"] = df.whole_area - df.house_area

    df = df[["countyid","latitude","longitude","yearbuilt","bathroomcnt","bedroomcnt","house_area", "land_area", "whole_area", "house_value","land_value","whole_value","taxrate","logerror", "month"]]
    return df

def define_month(df):
    """
    Replace the transactiondate with the month of the transaction.
    """
    df["month"] = pd.DatetimeIndex(df.transactiondate).month
    df.drop("transactiondate",axis=1, inplace=True)
    return df

def encode_hot(train, test, col_name):
    encoded_values = sorted(list(train[col_name].unique()))

    # Integer Encoding
    int_encoder = LabelEncoder()
    train.encoded = int_encoder.fit_transform(train[col_name])
    test.encoded = int_encoder.transform(test[col_name])

    # create 2D np arrays of the encoded variable (in train and test)
    train_array = np.array(train.encoded).reshape(len(train.encoded),1)
    test_array = np.array(test.encoded).reshape(len(test.encoded),1)

    # One Hot Encoding
    ohe = OneHotEncoder(sparse=False, categories='auto')
    train_ohe = ohe.fit_transform(train_array)
    test_ohe = ohe.transform(test_array)

    # Turn the array of new values into a data frame with columns names being the values
    # and index matching that of train/test
    # then merge the new dataframe with the existing train/test dataframe
    train_encoded = pd.DataFrame(data=train_ohe, columns=encoded_values, index=train.index)
    train = train.join(train_encoded)

    test_encoded = pd.DataFrame(data=test_ohe, columns=encoded_values, index=test.index)
    test = test.join(test_encoded)

    return train, test