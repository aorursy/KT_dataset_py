import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.cluster import KMeans
def photo(image):

    img=mpimg.imread(image)

    #imgplot = plt.imshow(img)

    scaled = img / 255

    data = scaled.reshape(scaled.shape[0] * scaled.shape[1], scaled.shape[2])

    kmeans = KMeans(4)

    kmeans.fit(data)

    labels = kmeans.predict(data)

    new_colors = kmeans.cluster_centers_[kmeans.labels_]

    recolored = new_colors.reshape(img.shape)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    fig.subplots_adjust(wspace=0.05)

    ax[0].imshow(img)

    ax[0].set_title('Original Image', size=16)

    ax[1].imshow(recolored)

    ax[1].set_title('4-color Image', size=16);



    ax[0].set_xticks([])

    ax[0].set_yticks([])

    ax[1].set_xticks([])

    ax[1].set_yticks([])

    plt.imsave(f'4_{image}', recolored)