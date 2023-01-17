import glob

import itertools

import collections



from PIL import Image as IMG

import cv2

from tqdm import tqdm_notebook as tqdm

import pandas as pd

import numpy as np

import torch

import imagehash

from joblib import Parallel, delayed

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

pd.options.display.max_rows = 20

pd.options.display.max_columns = 1000
train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')

def change_PetID(x):

    err = 9-len(x['PetID'])

    if err > 0:

        x['PetID'] = '0'*err + x['PetID']

    return x

train = train.apply(change_PetID,axis=1)

test = test.apply(change_PetID,axis=1)

train.loc[:,'Category'] = 'train'

test.loc[:,'Category'] = 'test'

test.loc[:,'AdoptionSpeed'] = np.nan

df = pd.concat([train, test], sort=False).reset_index(drop=True)
def show_all_sim_pics(a,b):

    def tt(a):

        if a in list(train['PetID']):

            return 'train'

        else:

            return 'test'

    

    sa = df[(df['PetID']==a)].iloc[0].loc['PhotoAmt']

    sb = df[(df['PetID']==b)].iloc[0].loc['PhotoAmt']

    print(tt(a),a,sa)

    print(tt(b),b,sb)

    for i in range(1,np.max((int(sa)+1,int(sb)+1))):

        fig = plt.figure(figsize=(10, 20))

        fig.add_subplot(1,2,1)

        if i <= sa:

            image1 = cv2.imread('../input/%s_images/%s-%s.jpg' % (tt(a), a, str(i)))

            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

            plt.imshow(image1)

        else:

            plt.imshow([[0]])

        fig.add_subplot(1,2, 2)

        if i <= sb:

            image2 = cv2.imread('../input/%s_images/%s-%s.jpg' % (tt(b), b, str(i)))

            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            plt.imshow(image2)

        else:

            plt.imshow([[0]])

        plt.show()

    return df[(df['PetID'] == a) | (df['PetID'] == b)]



        

interesting_pairs = (('0940ecf2b', '937c2e14f'), 

                     ('6d5afdb0f', '95b94b4ac'), 

                     ('23379775f', '6b446b436'), 

                     ('1dfcfe139', '4dc36183c'), 

                     ('89c8b19eb', 'dc99804de'), 

                     ('bc2272277', 'eedb57cd6'), 

                     ('5ed5663ae', '8673a672d'), 

                     ('75b33cd1d', 'e93b2f0c5'), 

                     ('0c859ef92', '23829dc63'), 

                     ('5417ca8e6', 'cc563b215'), 

                     ('2d8db1c19', '4733d5e7a'), 

                     ('58e8cb9e1', 'afda0a2dd'), 

                     ('29558e678', 'd9801429b'), 

                     ('3721ed900', '9b9e24b37'), 

                     ('1bff3e46f', '65650b974'), 

                     ('40dfeeb65', '8d32f11b0'), 

                     ('2a9c5eab2', 'c24f53ec1'), 

                     ('0a98163c8', '0f9697729'), 

                     ('1b6f92bc2', '6cfba936a'), 

                     ('14c70b2c2', '759e2e94e'), 

                     ('59e7b5f52', '65650b974'), 

                     ('4733d5e7a', 'de591574b'), 

                     ('87b753139', 'e99a775d2'), 

                     ('9ab460730', 'de0ace433'))

show_all_sim_pics(*interesting_pairs[0])
show_all_sim_pics(*interesting_pairs[1])
show_all_sim_pics(*interesting_pairs[2])
show_all_sim_pics(*interesting_pairs[3])
show_all_sim_pics(*interesting_pairs[4])
show_all_sim_pics(*interesting_pairs[5])
show_all_sim_pics(*interesting_pairs[6])
show_all_sim_pics(*interesting_pairs[7])
print(len(interesting_pairs))

counter = collections.Counter()

for petid1, petid2 in interesting_pairs:

    row1 = df.loc[df['PetID']==petid1].iloc[0]

    row2 = df.loc[df['PetID']==petid2].iloc[0]

    for attr in train.columns:

        if getattr(row1, attr) != getattr(row2, attr):

            counter[attr] += 1        

counter