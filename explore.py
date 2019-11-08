import pandas as pd 
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans, dbscan
from scipy import stats

def features_num_values(df):
    """
    Takes each numeric feature in dataframe and plots the distribution and display the value count.
    """
    features_num = list(df.select_dtypes(np.number).columns)

    for feature in features_num:
        sns.distplot(df[feature])
        plt.title(f'Distribution plot and value counts for {feature}')
        plt.yticks([])
        plt.show()
        if df[feature].nunique() >= 25:
            print(df[feature].value_counts(bins=10, sort=False))
        else:
            print(df[feature].value_counts())

def elbow(df, points=10):
    ks = range(1,points+1)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k')
    plt.show()


def k_cluster_2d(df, x, y, n_max, n_min=2):
    """
    Plots a 2D cluster map of an inputted x and y, starting at 2 clusters, up to inputted max cluster amount
    Import whole dataframe, select the x and y values to cluster.
    """
    for n in range(n_min,n_max+1):
        if "cluster" in df.columns:
            df = df.drop("cluster",axis=1)
        kmeans = KMeans(n_clusters=n, random_state=123)
        kmeans.fit(df)
        df["cluster"] = kmeans.predict(df)
        df.cluster = 'cluster_' + (df.cluster + 1).astype('str')

        sns.relplot(data=df, x=x, y=y, hue='cluster')
        plt.title(f'{n} Clusters')


def k_cluster_3d(df, x, y, z, n):
    """
    Displays 3d plot of clusters.
    """
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(df)
    labels = kmeans.labels_

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df[x], df[y], df[z], c=labels, edgecolors="grey",alpha=.5)
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    ax.set(xlabel=x, ylabel=y, zlabel=z)
    ax.xaxis.labelpad=-5
    ax.yaxis.labelpad=-5
    ax.zaxis.labelpad=-5
    plt.show()

def k_cluster_all(df, x, n):
    """
    Takes a dataframe and a single feature, and performs a 2d kmeans clustering on that feature against all other features in the dataframe. Also, specify the number of clusters to explore.
    """
    if "cluster" in df.columns:
        df = df.drop("cluster",axis=1)
    
    kmeans = KMeans(n_clusters=n, random_state=123)
    kmeans.fit(df)
    df["cluster"] = kmeans.predict(df)
    df.cluster = 'cluster_' + (df.cluster + 1).astype('str')

    for col in df.columns:
        sns.relplot(data=df, x=x, y=col, hue='cluster')
        plt.show()


def db_cluster_2d(df, eps, minPts):
    """
    Import a dataframe containing only the selected x and y value in it, and the eps and min points
    """
    cores, labels = dbscan(df, eps=eps, min_samples=minPts)
    df['cluster'] = labels

    for c in df.cluster.unique():
        subset = df[df.cluster == c]
        marker = 'x' if c == -1 else 'o'
        plt.scatter(subset.iloc[:,0], subset.iloc[:,1], marker=marker, label=c, zorder=3)
        
    plt.title(f'DBSCAN, eps={eps}, minPts={minPts}')
    plt.legend()


def stat_cluster()
    for df.cluster.unique()
    stats.ttest_1samp(train_ex.logerror[train_ex.cluster_loc == 4],train_ex.logerror.mean())


