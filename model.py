import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import median_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def X_and_y(train, test):
    X_train = train.drop(columns="logerror")
    X_test = test.drop(columns="logerror")
    y_train = train[["logerror"]]
    y_test = test[["logerror"]]
    return X_train, X_test, y_train, y_test


def compute_baseline(df):
    df["logerror_mean"] = df.logerror.mean()
    return df


def kmeans_model(df,n):
    kmeans = KMeans(n_clusters=n, random_state=123)
    kmeans.fit(df)
    return kmeans.labels_


def linear_model(X_train, y_train, df):
    lm=LinearRegression()
    lm.fit(X_train,y_train)
    lm_predictions=lm.predict(X_train)
    df['lm']=lm_predictions
    return df, lm


def evaluate(actual, predict):
    mae = median_absolute_error(actual,predict)
    r2 = r2_score(actual, predict)
    return mae, r2 


def plot_linear_model(actuals, lm, baseline):
    plot = pd.DataFrame({'actual': actuals,
                'linear model': lm,
                'baseline': baseline.ravel()})\
    .melt(id_vars=['actual'], var_name='model', value_name='prediction')\
    .pipe((sns.relplot, 'data'), x='actual', y='prediction', hue='model', alpha=.5)

    plt.plot([actuals.min(),actuals.max()],[lm.min(),lm.max()], \
            c='black', ls=':', linewidth = 3, alpha=.9)
 
    plt.ticklabel_format(style="plain")
    plt.ylabel("Predicted")
    plt.xlabel("Actuals")
    plt.title("How does the liner regression model compare to the mean log error?")
    plt.show()
