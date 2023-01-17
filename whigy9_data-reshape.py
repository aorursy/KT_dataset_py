import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/train.csv')

df.head(2)
import json

import ast

from collections import defaultdict



JSON_COLUMNS = ['belongs_to_collection', 'genres', 'production_companies', 'production_countries', 'spoken_languages',

                'Keywords', 'cast', 'crew']

DTYPE = {}



def load_df(csv_path=''):

    

    # define converters

    def convert_json(series):

        if series == "" or series == "[]" or series == "#N/A":

            return {}

        try:

            data_list = ast.literal_eval(series)

            if len(data_list) == 1:

                return data_list[0]

            else:

                dd = defaultdict(list)

                for d in data_list:

                    for key, value in d.items():

                        dd[key].append(value)

                return dict(dd)

        except Exception as e:

            print(series)

            raise



    df = pd.read_csv(csv_path, 

                     converters={column: convert_json for column in JSON_COLUMNS})

    

    for column in JSON_COLUMNS:

        normalized = df[column].apply(pd.Series)

        new_cols = dict(zip(normalized.columns, ("{}_{}".format(column, c) for c in normalized.columns)))

        normalized= normalized.rename(columns=new_cols)

        df = pd.concat([df.drop([column], axis=1), normalized], axis=1)



    return df

df2 = load_df('../input/train.csv')
df2.head(3)
#df2.to_csv("data/clean_train_data.csv", index= False)