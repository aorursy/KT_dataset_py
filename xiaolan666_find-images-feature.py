# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

from pandas import DataFrame

from PIL import Image

import json

import cv2

%matplotlib inline
print(os.listdir("../input"))
train = pd.read_csv('../input/train/train.csv')
train.drop('Description',axis=1).head()
cat = train[train['Type']==2].drop(['Description','RescuerID','PetID'],axis=1)
cat.head()
cat['AdoptionSpeed'].value_counts().sort_index().plot('bar', color='teal')

plt.title('Adoption speed classes counts')
dog = train[train['Type']==1].drop(['Description','RescuerID','PetID'],axis=1)
dog.head()
dog['AdoptionSpeed'].value_counts().sort_index().plot('bar', color='teal')

plt.title('Adoption speed classes counts')
test = pd.read_csv('../input/test/test.csv')
all_data = pd.concat([train, test])
all_data.head()
def convert_image(all_data,Type):

    images = [i.split('-')[0] for i in os.listdir('../input/train_images/')]

    size_dict = {1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large'}

    for t in all_data['Type'].unique():

        for m in all_data['MaturitySize'].unique():

            df = all_data.loc[(all_data['Type'] == t) & (all_data['MaturitySize'] == m)]

            top_breeds = list(df['Breed1'].value_counts().index)[:5]

            m = size_dict[m]

            print(f"Most common Breeds of {m} {t}s:")



            fig = plt.figure(figsize=(25, 4))



            for i, breed in enumerate(top_breeds):

                # excluding pets without pictures

                b_df = df.loc[(df['Breed1'] == breed) & (df['PetID'].isin(images)), 'PetID']

                if len(b_df) > 1:

                    pet_id = b_df.values[1]

                else:

                    pet_id = b_df.values[0]

                ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])

                #im = Image.open("../input/train_images/" + pet_id + '-1.jpg')

                im = cv2.imread("../input/train_images/" + pet_id + '-1.jpg')

                im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

                if(Type=='gray'):

                    im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)

                    

                if(Type=='threshold'):

                    gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)

                    ret,im=cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

                plt.imshow(im)

                ax.set_title(f'Breed: {breed}')

            plt.show();
convert_image(all_data,'ori')
convert_image(all_data,'threshold')