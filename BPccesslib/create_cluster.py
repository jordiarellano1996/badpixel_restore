"""
Created on Thu Nov  4 17:29:36 2021

@author: titoare
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from get_KMean_k import chooseBestKforKMeans, get_clusters


def create_mask_px(img_data, seed, dead_p=[0.9, 0.1]):
    np.random.seed(seed)
    img_shape = img_data.shape
    mask_px = np.random.choice(a=[False, True], size=img_shape, p=dead_p)
    return mask_px


def main(img_df, mask_px, sensibility=0.003, plt_elbow=False):
    """Create masked pixels invert"""
    inv_mask_px = np.invert(mask_px)

    """Escale data"""
    mms = MinMaxScaler()
    scaled_data = mms.fit_transform(img_df.values)
    del mms

    """Mask pixels error value"""
    scaled_data = np.where(inv_mask_px, scaled_data, 255)

    """ Optimum clusters """
    K = range(1, 10)
    data_in = np.expand_dims(scaled_data.reshape(-1, 1)[inv_mask_px.reshape(-1, 1)], axis=1)
    best_k, results = chooseBestKforKMeans(data_in, K, sensibility)
    print(f"The optimum clusters are: {best_k}")
    if plt_elbow:
        sns.lineplot(x=K, y=results.values.ravel(), marker="o")
        plt.show()

    """Get clusters"""
    cluster_data = get_clusters(scaled_data, best_k)
    error_value_cd = cluster_data[mask_px][0]

    return cluster_data, error_value_cd


if __name__ == "__main__":
    """ Import data"""
    img_df = pd.read_csv("../data/image_correct.csv", index_col=0)
    # img1_df = pd.read_csv("../data/image1_correct.csv", index_col=0)
    # img2_df = pd.read_csv("../data/image2_correct.csv", index_col=0)
    plt.imshow(img_df.values, cmap="inferno")
    plt.show()
    # plt.imshow(img1_df.values, cmap="inferno")
    # plt.show()
    # plt.imshow(img2_df.values, cmap="inferno")
    # plt.show()

    """Create masked pixels"""
    mask_px = create_mask_px(img_df.values, 2022)
    # %% Run code here in cell foramt
    """Fill dead pixels"""
    import time
    t = time.time()
    cluster_data, error_value_cd = main(img_df, mask_px, plt_elbow=True)
    print(f"Time {time.time() - t}")
    del t
