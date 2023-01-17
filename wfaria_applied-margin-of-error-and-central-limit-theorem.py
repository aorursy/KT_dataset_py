# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def get_opinion_from_random_person():

    # Usually this probabiliy is not known. Setting it here for simulation purposes.

    probability_of_alice_win = 0.53

    return np.random.binomial(

        n = 1,

        p = probability_of_alice_win)



get_opinion_from_random_person()
def get_sample_from_distribution(n):

    """

    Creates an array of size n.

    Each value will be 1 if some person would vote on Alice and

    0 if he would vote on Bob.

    """

    sample_opinions = []

    for i in range(n):

        sample_opinions.append(get_opinion_from_random_person())

    

    return sample_opinions



example = get_sample_from_distribution(10)

print(example)
def get_n_samples_from_distribution(samples_number, sample_size):

    """

    Get multiple samples from our target 'unknown distribution'.

    """

    people_per_day = [sample_size] * samples_number

    week_samples = list(map(get_sample_from_distribution, people_per_day))

    return week_samples



def get_n_sample_means_from_distribution(samples_number, sample_size):

    samples = get_n_samples_from_distribution(

        samples_number = samples_number,

        sample_size = 100)



    return list(map(np.mean, samples))



sample_means = get_n_sample_means_from_distribution(5, 100)

print(sample_means)
fig, axes = plt.subplots(2,3, figsize = (12, 8))

fig.subplots_adjust(hspace=0.4, wspace=0.3)

axes = axes.ravel()



days = [5, 50, 500, 1000, 10000, 100000]

for i in range(len(days)):

    sample_means = get_n_sample_means_from_distribution(

        samples_number = days[i],

        sample_size = 100)    

    axes[i].hist(sample_means, bins=30)

    axes[i].set_title("{0} days".format(days[i]))
fig, axes = plt.subplots(2,3, figsize = (12, 8))

fig.subplots_adjust(hspace=0.4, wspace=0.3)

axes = axes.ravel()



sample_size = [10, 100, 10000, 1000000, 10000000, 100000000000000]

for i in range(len(sample_size)):

    sample_means = get_n_sample_means_from_distribution(

        samples_number = 100,

        sample_size = sample_size[i])    

    axes[i].hist(sample_means, bins=30)

    axes[i].set_title("Sample size: {0}".format(sample_size[i]))
def get_margin_of_error_interval(sample_size, number_of_std_dvt):

    day_sample = get_n_samples_from_distribution(

        samples_number = 1,

        sample_size = sample_size)

    sample_mean = np.mean(day_sample)

    sample_std_dvt = np.std(day_sample)

    estimated_std_dvt = sample_std_dvt / math.sqrt(sample_size)

    margin_of_error = number_of_std_dvt * estimated_std_dvt

    return { "error": margin_of_error, "mean": sample_mean }



get_margin_of_error_interval(100, 2)
x = []

y = []

y1 = []

y2 = []



fig, axes = plt.subplots(1, 2, figsize = (10, 6))

axes = axes.ravel()

for power_of_ten in range(1, 8):

    x.append(power_of_ten)

    y.append(get_margin_of_error_interval(10 ** power_of_ten, 2))

    

for y_aux in y:

    y1.append(y_aux["error"])

    y2.append(y_aux["mean"])



axes[0].set_title("Margin of error X Log of Sample Size (Smaller is better)")

axes[0].set_ylabel("Margin of error")

axes[0].set_xlabel("Sample Size = 10^x")

axes[0].plot(x, y1)



axes[1].set_title("Sample Mean X Log of Sample Size")

axes[1].set_ylabel("Sample Mean")

axes[1].set_xlabel("Sample Size = 10^x")

axes[1].plot(x, y2)
get_margin_of_error_interval(sample_size = 10000, number_of_std_dvt = 2)
0.498 / (3010) ** (1/2)
0.00907707857637729 * 1.96
