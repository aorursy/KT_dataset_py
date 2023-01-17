# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.



data = pd.read_csv("../input/train.tsv", sep = '\t')



data.head()



data['item_condition_id'].value_counts()
# understanding relationship between item_condition_id and price

import matplotlib.pyplot as plt

plt.ylim(-50, 200)

import seaborn as sns

sns.violinplot(data=data, x = 'item_condition_id', y = 'price')



# price distribution

plt.xlim(-50, 300)

sns.distplot(data['price'])



# mean price of popular brands

brands = data['brand_name'].value_counts()[:10].index

brands

k = data.groupby('brand_name').mean()['price']

k

p = k.loc[brands]

fig, ax = plt.subplots(figsize=(20, 10))

plt.ylabel('Mean price')

plt.bar(p.index, p)
# NLP techniques

import nltk

from nltk.tokenize import word_tokenize



# function to count adjectives

def adj_count( text ) :

    words = word_tokenize(text)

    k =  nltk.pos_tag(words)

    y = [ x[1] for x in k ]

    return y.count('JJ')

    

 # function to counts rb   

def rb_count( text ) :

    words = word_tokenize(text)

    k =  nltk.pos_tag(words)

    y = [ x[1]  for x in k]

    return y.count('RB')

    

    

data['item_description'] = data['item_description'].astype(str)    



fnc = lambda x: adj_count(x)

#fncc = lambda x: x.decode('utf-8')

#pp = pd.DataFrame()

#data['item_description'] = data['item_description'].apply(fncc)

#pp['adj_count'] = data['item_description'].apply(fnc)
#import pandas as pd

test = pd.read_csv("../input/test.tsv", sep='\t')

test.head()

test['item_description'] = test['item_description'].astype(str)  

kk = pd.DataFrame()

kk['adj_count'] = test['item_description'].apply(fnc)



#pp.to_csv("train_adj.csv", index=False)

kk.to_csv("test_adj.csv", index=False)

# no. of words in description

#data['word_count'] = data['item_description'].apply(lambda x: len(x.split()))

# no. of characters in description

#data['char_count'] = data['item_description'].apply(lambda x: len(x))



#data.head()

# inverse characters per word in item description

#data['mean_desc'] = data['item_description'].apply(lambda x: float( len(x.split()))/ len(x))

#data.head()
# words in name

#data['words_in_name'] = data['name'].apply(lambda x: len(x.split()))

#data.head()
# no. of characters in name

#data['char_count_name'] = data['name'].apply(lambda x: len(x))



# inverse of characters per word in item name



#data['mean_name'] = data['name'].apply(lambda x: float( len(x.split()))/ len(x))

#data.head()
# missing value imputation

#data.isnull().sum()
#data['category_name'].value_counts()
# filling missing category with the higest occuring category

#data['category_name'].fillna('Women/Athletic Apparel/Pants, Tights, Leggings', inplace=True)

#data['category_name'].isnull().sum()
# filling missing values for brand name

#data['brand_name'].value_counts()
# filling missing values with 'pink'

#data['brand_name'].fillna('PINK', inplace=True)

#data['brand_name'].isnull().sum()
#  label encoding

#l = ['name', 'category_name', 'brand_name', 'item_description']



#from sklearn import preprocessing

#le = preprocessing.LabelEncoder()



#for x in l:

  #  le.fit(data[x])

 #   data[x] = le.transform(data[x])

#

##data.head()



# scaling

#data['mean_desc'] = data['mean_desc']*10

#data['mean_name'] = data['mean_name'] * 10
# attributes as x and label as y

#y = data[['price']]

#x = data[['item_condition_id', 'category_name','brand_name', 'shipping',  'adj_count', 'word_count', 'char_count', 'mean_desc', 'words_in_name', 'char_count_name', 'mean_name']]

#x.head()
# splitting into train and test

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#y_train
# training with gradient boosting regressor

# importing model

#from sklearn import ensemble

#clf = ensemble.GradientBoostingRegressor(learning_rate=0.3, n_estimators=1200, max_depth = 3,warm_start = True, verbose=1, random_state=45)



# training model

#clf.fit(X_train, y_train.values.ravel())
# testing on test data

#from sklearn.metrics import mean_squared_error

#mse = mean_squared_error(y_test, clf.predict(X_test))

#print("MSE: %.4f" % mse)
