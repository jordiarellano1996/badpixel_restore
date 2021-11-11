"""
Created on Wed Nov  3 20:00:36 2021

@author: titoare
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def create_elbow(data_in, _range=15):
    sum_of_squared_distances = []
    K = range(1, _range)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_in.reshape(-1, 1))
        """Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided."""
        sum_of_squared_distances.append(km.inertia_)

    return K, sum_of_squared_distances


def get_clusters(data_in, k):
    km = KMeans(n_clusters=k)
    out = km.fit_predict(data_in.reshape(-1, 1))
    return out.reshape(data_in.shape)


def kMeansRes(scaled_data, k, alpha_k=0.02):
    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns 
    -------
    scaled_inertia: float
        scaled inertia value for current k           º
    '''

    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia


def chooseBestKforKMeans(scaled_data, k_range, alpha_k_in):
    ans = []
    for k in k_range:
        scaled_inertia = kMeansRes(scaled_data, k, alpha_k=alpha_k_in)
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns=['k', 'Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results


if __name__ == "__main__":
    """ Import data"""
    img_df = pd.read_csv("../data/image_correct.csv", index_col=0)
    plt.imshow(img_df.values, cmap="inferno")
    plt.show()

    """Visualazing elbow and deciding K"""
    img_in = img_df.values
    K, sum_of_squared_distances = create_elbow(img_in, _range=14)
    sns.lineplot(x=K, y=sum_of_squared_distances, marker="o")
    plt.show()

    """Escale data"""
    mms = MinMaxScaler()
    scaled_data = mms.fit_transform(img_in.reshape(-1, 1))

    """Choose best value on KMeans and plot"""
    K = range(1, 14)
    best_k, results = chooseBestKforKMeans(scaled_data, K, 0.005)
    sns.lineplot(x=K, y=results.values.ravel(), marker="o")
    plt.show()

    """Grouping clusters in matrix"""
    clusters_mx = get_clusters(img_df.values, best_k)
