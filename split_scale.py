import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

# split_my_data(X, y, train_pct, random_state=123)
def split_my_data(df, train_pct, random_state=123):
    train, test = train_test_split(df, train_size=train_pct, random_state=random_state)
    return train, test

def transform_scaler(train, test, scaler):  
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return train_scaled, test_scaled

# standard_scaler()
def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled, test_scaled = transform_scaler(train, test, scaler)

    return train_scaled, test_scaled, scaler

# scale_inverse()
def scale_inverse(train_scaled, test_scaled, scaler):
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
 
    return train_unscaled, test_unscaled

# uniform_scaler()
def uniform_scaler(train, test):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)

    train_scaled, test_scaled = transform_scaler(train, test, scaler)

    return train_scaled, test_scaled, scaler   

# gaussian_scaler()
def gaussian_scaler(train, test):
    scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)

    train_scaled, test_scaled = transform_scaler(train, test, scaler)

    return train_scaled, test_scaled, scaler
 
# min_max_scaler()
def min_max_scaler(train, test):
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    train_scaled, test_scaled = transform_scaler(train, test, scaler)

    return train_scaled, test_scaled, scaler
 
# iqr_robust_scaler()
def iqr_robust_scaler(train, test):
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)

    train_scaled, test_scaled = transform_scaler(train, test, scaler)

    return train_scaled, test_scaled, scaler  

