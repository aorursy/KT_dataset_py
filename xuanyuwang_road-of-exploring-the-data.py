import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import Counter

from subprocess import check_output



train = pd.read_csv("../input/train.csv").as_matrix()

test = pd.read_csv("../input/test.csv").as_matrix()

x_train = train[0:, 1:]

y_train = train[0:, 0]

x_test = test
numOfImage = len(y_train)

imageOfDigit = dict()

for i in range(numOfImage):

    label = y_train[i]

    if label in imageOfDigit.keys():

        imageOfDigit[label].append(x_train[i])

    else:

        imageOfDigit[label] = [x_train[i]]



mean = dict()

for label in imageOfDigit.keys():

    length = len(imageOfDigit[label])

    imageOfDigit[label] = np.array(imageOfDigit[label])

    mean[label] = np.mean(imageOfDigit[label], axis=0)

    
def plot_image(indexOfDigits, size=28, title="index"):

    plt.figure(figsize=(12, 12))

    plt.gray()

    for label in indexOfDigits.keys():

        plt.subplot(4, 3, label + 1)

        plt.imshow(indexOfDigits[label].reshape(size, size))

        plt.title("{} of {}".format(title, label))

    plt.show()
plot_image(mean, title="mean")
predictions = []

for case in x_train:

    distance = []

    # Calculate the distances from mean shapes of 0-9

    for i in range(10):

        d = ((mean[i] - np.array(case))**2).sum()

        distance.append(d)

    # Choose the cloest distance and use the corresponding digit as prediction

    closest = min(distance)

    for i in range(10):

        if distance[i] == closest:

            predictions.append(i)

            break
# statistic the information of number of right predictions and wrong predictions

right_count = 0

wrongs = dict()

for i in range(numOfImage):

    # For right predictions, we just count the number

    if predictions[i] == y_train[i]:

        right_count += 1

    # For wrong predictions, we also need to know what is the right answer

    else:

        if y_train[i] in wrongs.keys():

            wrongs[y_train[i]].append(predictions[i])

        else:

            wrongs[y_train[i]] = [predictions[i]]

accuracy = right_count / numOfImage

print("Accuracy: {}".format(accuracy))
wrongMat = [[0 for _ in range(10)] for _ in range(10)]

for i in range(10):

    s = Counter(wrongs[i])

    for j in range(10):

        wrongMat[i][j] = s[j]

plt.figure(figsize=(10, 10))

plt.imshow(wrongMat)

plt.title("Fig. 2 Wrong Answer Matrix")

plt.show()
std = dict()

for i in range(10):

    std[i] = np.std(imageOfDigit[i], axis=0)

plot_image(std, title="std")
median = dict()

for i in range(10):

    median[i] = np.median(imageOfDigit[i], axis=1)
plot_image(std, title="median")