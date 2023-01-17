import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import itertools





# to read json file

# http://stackoverflow.com/questions/34684468/read-json-file-using-pandas

train_data=pd.read_json('../input/train.json')
#Exercise (b)



print("How many samples (dishes) are there in the training set?")

print(len(pd.Series(train_data['id'])))



print('How many categories (types of cuisine)?')

print(pd.Series(train_data['cuisine']).value_counts())



print('Use a list to keep all the unique ingredients appearing in the training set. How many unique ingredients are there?')

ingredients_list=list(itertools.chain(*train['ingredients']))

print(pd.Series(ingredients_list).value_counts().size)