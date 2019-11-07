import pandas as import pd 
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt

def k_cluster_2d(df, x, y):
    """
    Import whole dataframe, and selected x and y values to compare on a plot
    """
    kmeans = KMeans(n_clusters=3, random_state=123)
    kmeans.fit(df)
    df["cluster"] = kmeans.predict(df)

    sns.relplot(data=df, x=x, y=y, hue='cluster')


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



