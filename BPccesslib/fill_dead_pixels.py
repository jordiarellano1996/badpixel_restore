"""
Created on Tue Nov  9 10:02:52 2021

@author: titoare
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from create_cluster import main, create_mask_px
from sklearn.metrics import mean_squared_error
import cv2


def get_window(image_in, center_in, pad_in):
    (y, x) = center_in
    return image_in[y - pad_in:y + pad_in + 1].T[x - pad_in:x + pad_in + 1].T


def sqr_err(y_true, y_pred):
    """

    :param y_true: true values of y
    :param y_pred: predicted values of y
    :return: array of lenght original data containing mean squared error for each predictions
    """
    if len(y_true) != len(y_pred):
        raise IndexError("Mismathced array sizes, you inputted arrays with sizes {} and {}".format(len(y_true),
                                                                                                   len(y_pred)))
    else:
        length = len(y_true)

    sqrerror_out = [(y_pred[i] - y_true[i]) ** 2 for i in range(length)]

    return np.array(sqrerror_out)


def plot_sqr_err(window_center_in, data_mean_in, msq_err_in):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax2 = ax.twinx()
    x_plot = np.arange(0, len(window_center_in))
    sns.scatterplot(x=x_plot, y=window_center_in, ax=ax, color='blue')
    sns.scatterplot(x=x_plot, y=data_mean_in, ax=ax2, color='orange')
    ax.set_title(f'Mean squared error: {msq_err_in}', loc='left', fontweight='bold')
    ax.legend(['real'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax2.legend(["pred"], bbox_to_anchor=(1.05, 0.96), loc=2, borderaxespad=0.)
    plt.show()


if __name__ == "__main__":

    """ Import data"""
    img_df = pd.read_csv("../data/image_correct.csv", index_col=0)
    plt.imshow(img_df.values, cmap="inferno")
    plt.show()

    """Create masked pixels"""
    mask_px = create_mask_px(img_df.values, 2022, dead_p=[0.99, 0.01])

    """Cluster the data"""
    clust_data, error_value_cd = main(img_df, mask_px, plt_elbow=True)

    """Add BORDER to all matrix"""
    pad = 5
    if pad % 2 == 0:
        pad = pad - 1

    image = img_df.values.copy()
    image_bor = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    mask_px_bor = mask_px.astype(np.int8)
    mask_px_bor = cv2.copyMakeBorder(mask_px_bor, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=0)
    clust_data_bor = cv2.copyMakeBorder(clust_data, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT,
                                        value=int(error_value_cd))

    """ Get bad pixels position"""
    bp_pos = np.where(mask_px_bor == 1)
    bp_pos = np.array(list(zip(bp_pos[0], bp_pos[1])))

    """Fill dead pixels"""
    data_mean = []
    window_center = []

    for center in bp_pos:
        window_data = get_window(image_bor, center, pad)
        window_cluster = get_window(clust_data_bor, center, pad)
        window_mask_px = get_window(mask_px_bor, center, pad)
        window_mask_px = np.invert(np.array(window_mask_px).astype(np.bool))

        _mean = np.average(window_data, weights=window_mask_px)
        data_mean.append(_mean)
        window_center.append(window_data[pad][pad])

        image[center[0] - pad][center[1] - pad] = _mean

    del window_cluster, window_data, window_mask_px, center, pad, _mean

    msq_err = mean_squared_error(window_center, data_mean)
    plot_sqr_err(window_center, data_mean, msq_err)
    print(f"Mean squared error: {msq_err}")
