import numpy as np

import os

import pickle

import gc #garabag collection

import xgboost as xgb

import re

import pandas as pd

from sklearn.model_selection import train_test_split



#max_num_features = 10

#pad_size = 1

#boundary_letter = -1

#space_letter = 0

max_data_size = 960000

max_num_features = 10



def ascii(x, max_num_features = 10, space_letter = 0):

    try:

        t = map(ord, x[0])

    except:

        return max_num_features*[space_letter] + [-1]

    l = min(len(t), max_num_features)

    return t[:l] + (max_num_features-l)*[space_letter] + [x[-1]]



def context_window(data, pad_pre = 1, pad_pos = 1, boundary_letter = -1):

    #pad_before: num of words before

    new_data = []

    pad = max_num_features*[0] + [-1]

    emp = [boundary_letter]+(max_num_features)*[0]

    data = [pad for i in range(pad_pre)] + data + [pad for i in range(pad_pos)]

    for i in range(pad_pre, len(data) - pad_pos):

        l = data[i][-1]

        t, tmp = [], []

        for j in range(i-pad_pre, i+pad_pos+1):

            if data[j][boundary_letter] == l:

                tmp += [j]

                t += [boundary_letter] + data[j][:-1]

        new_data.append(emp*(tmp[0]-i+pad_pre) + t + emp*(i+pad_pos-tmp[-1])+[boundary_letter])

    return new_data



out_path = r'.'

df = pd.read_csv(r'en_train.csv')

gc.collect()

#sentence_id + ascii

x_data = map(ascii, [[df['before'][i],df['sentence_id'][i]] for i in range(max_data_size)]) 

gc.collect()

x_data = x_data[:max_data_size]

x_data = np.array(context_window(x_data))



y_data =  pd.factorize(df['class'])

labels = y_data[1]

y_data = np.array(y_data[0][:max_data_size])