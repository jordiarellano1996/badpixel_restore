"""
Created on Wed Nov  3 19:58:47 2021

@author: titoare
"""

import numpy as np
import cv2


def convolve(image, clust_img, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    print(image.shape)
    print(kernel.shape)
    print(clust_img.shape)

    pad = (kW - 1) // 2

    # Here we are simply replicating the pixels along the border of the image,
    # such that the output image will match the dimensions of the input image.

    image = cv2.copyMakeBorder(image, pad, pad, pad,
                               pad, cv2.BORDER_REPLICATE)
    clust_img = cv2.copyMakeBorder(clust_img, pad, pad, pad,
                                   pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    i = 0
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # the Region of Interest (ROI) from the image.
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum() / np.sum(kernel > 0)
            output[y - pad, x - pad] = k
            image[y][x] = k
            if i == 0 or i == 150:
                print(roi)
                print(k)

            i += 1

    return output


def kernels():
    # construct average blurring kernels used to smooth an image
    smallBlur = np.ones((3, 3), dtype="float") * (1.0 / (3 * 3))
    largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
    # construct a sharpening filter
    sharpen = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
    # construct the Laplacian kernel used to detect edge-like
    # regions of an image
    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")
    # construct the Sobel x-axis kernel
    sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")
    # construct the Sobel y-axis kernel
    sobelY = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), dtype="int")

    # custom kernel
    custom_k = np.array((
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1]), dtype="int")

    # construct the kernel bank, a list of kernels we're going
    # to apply using both our custom `convole` function and
    # OpenCV's `filter2D` function
    kernelBank = (
        ("small_blur", smallBlur),
        ("large_blur", largeBlur),
        ("sharpen", sharpen),
        ("laplacian", laplacian),
        ("sobel_x", sobelX),
        ("sobel_y", sobelY),
        ("custokk", custom_k)
    )
    return kernelBank
