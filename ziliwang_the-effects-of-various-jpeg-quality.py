import tensorflow as tf

import cv2

import sys

from matplotlib import pyplot as plt
a = cv2.imread('../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0075663.jpg')

a = cv2.resize(a , (224,224), interpolation=cv2.INTER_AREA)
quals = [100, 98, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]

encodes = [cv2.imencode('.jpg', a, (cv2.IMWRITE_JPEG_QUALITY, i))[1].tostring() for i in quals]

decodes = [tf.io.decode_jpeg(i) for i in encodes]

size_of_origin = sys.getsizeof(encodes[0])

compress_rate = [sys.getsizeof(i)/size_of_origin for i in encodes]
k_low =110

k_size=5

plt.figure(figsize=(16, 6))

for i in range(12):

    plt.subplot(2, 6, i+1)

    plt.imshow(decodes[i][k_low:k_low+k_size, k_low:k_low+k_size, :])

    plt.title(f'q:{quals[i]} r:{compress_rate[i]: .3f}')
k_low =110

k_size=10

plt.figure(figsize=(16, 6))

for i in range(12):

    plt.subplot(2, 6, i+1)

    plt.imshow(decodes[i][k_low:k_low+k_size, k_low:k_low+k_size, :])

    plt.title(f'q:{quals[i]} r:{compress_rate[i]: .3f}')
k_low =110

k_size=20

plt.figure(figsize=(16, 6))

for i in range(12):

    plt.subplot(2, 6, i+1)

    plt.imshow(decodes[i][k_low:k_low+k_size, k_low:k_low+k_size, :])

    plt.title(f'q:{quals[i]} r:{compress_rate[i]: .3f}')
k_low =110

k_size=50

plt.figure(figsize=(16, 6))

for i in range(12):

    plt.subplot(2, 6, i+1)

    plt.imshow(decodes[i][k_low:k_low+k_size, k_low:k_low+k_size, :])

    plt.title(f'q:{quals[i]} r:{compress_rate[i]: .3f}')