# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import randrange



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





from csv import reader

from math import sqrt

 

# Load a CSV file

def load_csv(filename):

    dataset = list()

    with open(filename, 'r') as file:

        csv_reader = reader(file)

        for row in csv_reader:

            noval=0

            if not row:

                continue

            else:

                cc = 0

                for word in row:

                    if (word):

                        sword = str(word)

                    else:

                        noval =1

                        break

                    if (sword == '?'):

                        noval=1

                        break

                    elif (cc == 1):

                        if (sword == "Federal-gov"):

                            row[cc] = 8

                        elif (sword == "Local-gov"):

                            row[cc] = 6

                        elif(sword == "Never-worked"): 

                            row[cc] = 1

                        elif(sword == "Private"):

                            row[cc] = 5

                        elif(sword == "Self-emp-inc"):

                            row[cc] = 3

                        elif(sword == "Self-emp-not-inc"): 

                            row[cc] = 4

                        elif(sword == "State-gov"):

                            row[cc] = 7

                        elif(sword == "Without-pay"):

                            row[cc] = 2

                    elif (cc == 8):

                        if (sword == "Amer-Indian-Eskimo"):

                            row[cc] = 2

                        elif (sword == "Asian-Pac-Islander"):

                            row[cc] = 3

                        elif (sword == "Black"):

                            row[cc] = 1

                        elif (sword == "White"):

                            row[cc] = 4

                        elif (sword == "Other"):

                            row[cc] = 5

                    elif(cc == 9):

                        if (sword == "Female"):

                            row[cc] = 0

                        elif(sword == "Male"):

                            row[cc] = 1

                    elif (cc == 14):

                        if (sword == ">50K"):

                            row[cc] = 1

                        elif (sword == "<=50K"):

                            row[cc] = 0

                        else:

                            noval = 1

                            break

                    else:

                        word = None

                    cc += 1

                if (noval == 0):

                    dataset.append(row)

    return dataset



def drop_categorical_columns(dataset):

    dset = list()

    for row in dataset:

        cc = 0

        r = list()

        for c in row:

            if(cc == 2 or cc == 3 or cc == 5 or cc == 6 or cc == 7 or cc == 13):

                cc += 1

                continue

            r.append(c)

            cc += 1

        dset.append(r)

    return dset

 

# Convert string column to integer

def str_column_to_int(dataset, column):

    for row in dataset:

        cc = 0

        for c in row:

            row[cc] = int(row[cc])

            cc += 1

    return dataset

 

# Find the min and max values for each column

def dataset_minmax(dataset):

	minmax = list()

	for i in range(len(dataset[0])):

		col_values = [row[i] for row in dataset]

		value_min = min(col_values)

		value_max = max(col_values)

		minmax.append([value_min, value_max])

	return minmax

 

# Rescale dataset columns to the range 0-1

def normalize_dataset(dataset, minmax):

	for row in dataset:

		for i in range(len(row)):

			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

 

# Calculate the Euclidean distance between two vectors

def euclidean_distance(row1, row2):

	distance = 0.0

	for i in range(len(row1)-1):

		distance += (row1[i] - row2[i])**2

	return sqrt(distance)



def manhattan_distance(row1, row2): 

    _sum = 0

    for i in range(len(row1)-1):

        _sum += abs(row1[i] - row2[i])

    return _sum



def jaccard_similarity(row1, row2):

    r1 = []

    r2 = []

    for i in range(len(row1)-1):

        r1.append(row1[i])

    for i in range(len(row2)-1):

        r2.append(row2[i])

    intersection = len(list(set(r1).intersection(r2)))

    union = (len(r1) + len(r2)) - intersection

    return float(intersection) / union



# Locate the most similar neighbors

def get_neighbors(train, test_row, num_neighbors):

	distances = list()

	for train_row in train:

		dist = jaccard_similarity(test_row, train_row)

		distances.append((train_row, dist))

	distances.sort(key=lambda tup: tup[1])

	neighbors = list()

	for i in range(num_neighbors):

		neighbors.append(distances[i][0])

	return neighbors

 

# Split a dataset into k folds

def cross_validation_split(dataset, n_folds):

	dataset_split = list()

	dataset_copy = list(dataset)

	fold_size = int(len(dataset) / n_folds)

	for _ in range(n_folds):

		fold = list()

		while len(fold) < fold_size:

			index = randrange(len(dataset_copy))

			fold.append(dataset_copy.pop(index))

		dataset_split.append(fold)

	return dataset_split



# Calculate accuracy percentage

def accuracy_metric(actual, predicted):

	correct = 0

	for i in range(len(actual)):

		if actual[i] == predicted[i]:

			correct += 1

	return correct / float(len(actual)) * 100.0

    

# Evaluate an algorithm using a cross validation split

def evaluate_algorithm(dataset, algorithm, n_folds, *args):

	folds = cross_validation_split(dataset, n_folds)

	scores = list()

	for fold in folds:

		train_set = list(folds)

		train_set.remove(fold)

		train_set = sum(train_set, [])

		test_set = list()

		for row in fold:

			row_copy = list(row)

			test_set.append(row_copy)

			row_copy[-1] = None

		predicted = algorithm(train_set, test_set, *args)

		actual = [row[-1] for row in fold]

		accuracy = accuracy_metric(actual, predicted)

		scores.append(accuracy)

	return scores



# kNN Algorithm

def k_nearest_neighbors(train, test, num_neighbors):

	predictions = list()

	for row in test:

		output = predict_classification(train, row, num_neighbors)

		predictions.append(output)

	return(predictions)



# Make a prediction with neighbors

def predict_classification(train, test_row, num_neighbors):

	neighbors = get_neighbors(train, test_row, num_neighbors)

	output_values = [row[-1] for row in neighbors]

	prediction = max(set(output_values), key=output_values.count)

	return prediction

 

# Make a prediction with KNN on Iris Dataset

filename = '/kaggle/input/adult-census-income/adult.csv'

dataset = load_csv(filename)

dataset = drop_categorical_columns(dataset)



for i in range(len(dataset[0])):

    str_column_to_int(dataset, i)



num_neighbors = 5

n_folds = 5



# predict the label

scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)



print('Scores: %s' % scores)

print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))