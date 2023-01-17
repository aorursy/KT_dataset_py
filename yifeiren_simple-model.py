# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import gensim

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

from sklearn import ensemble

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score, average_precision_score, mean_squared_log_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
submit = True
df = pd.read_csv("../input/train.tsv", sep='\t')

if submit:

    df_submit = pd.read_csv("../input/test.tsv", sep='\t')
df.head(10)
df.info()
len(df.brand_name.unique())
len(df.item_condition_id.unique())
len(df.category_name.unique())
# list top catogries

df["category_name"].value_counts().head(20)
# list top brands

df["brand_name"].value_counts().head(20)
class Model:

    def __init__(self, model):

        self.model = model



    def train(self, X, y):

        self.model.fit(X, y)



    def predict(self, test_X):

        return self.model.predict(test_X)
# get top limit, mark the rest as other

def one_hot_encoding(df, col_name, limit=30):

    top = df[col_name].isin(df[col_name].value_counts().index[:limit])

    df_backup = df

    df.loc[~top, col_name] = "other" + col_name

    return pd.get_dummies(df, columns=[col_name])



def label_encoding(df, col_name):

    df[col_name] = df[col_name].astype('category')

    df[col_name] = df[col_name].cat.codes
# join train and test, then split them, so that they have the same encoding

price = df.pop('price')

df['is_submit'] = False

if submit:

    df_submit['is_submit'] = True

    combine = pd.concat([df, df_submit],axis = 0)

else:

    combine = df

label_encoding(combine, 'brand_name')

# label_encoding(combine, 'category_name')
sentences = []

for name in combine["category_name"].values:

    if type(name) != float:

        sentences.append(name.split("/"))

w2v_model = gensim.models.Word2Vec(sentences, size = 3, min_count=1, sg=1) # sg=1 use skip gram

n = 3

wv = [[] for i in range(0, n)] # shape 3 * n

for name in combine["category_name"].values:

    # initialize vector values

    word_vector = [0.0 for i in range(0,n)]

    if type(name) != float:

        for split in name.split("/"):

            # add word vectors

            word_vector += w2v_model[split]

    for i in range(0, n):

        wv[i].append(word_vector[i])

    

for i in range(0, n):

    col_name = "cat_name_" + str(i)

    combine[col_name] = wv[i]
if submit:

    df_submit = combine.loc[combine['is_submit']==True]

    

df = combine.loc[combine['is_submit']==False]
# features = ['brand_name', 'shipping', 'item_condition_id', 'category_name']

features = ['brand_name', 'item_condition_id', 'cat_name_0', 'cat_name_1', 'cat_name_2']

def train_and_test(model, df, price, fit):

    if not submit:

        # do not split train, test if submit

        X_train, X_test, y_train, y_test = train_test_split(df[features], price, test_size=0.25)

    else:

        X_train, y_train = df[features], price

    

    if fit: # for k nearest neghbour

        model.fit(X_train, y_train)

    else:

        model.train(X_train, y_train)

    if not submit:

        predictions = model.predict(X_test)

        print(np.sqrt(mean_squared_log_error(y_test.values, predictions)))
# model = Model(ensemble.RandomForestRegressor(n_estimators=10))

kn_model = KNeighborsRegressor(n_neighbors = 10)

train_and_test(kn_model, df, price, True)
if submit:

    predictions = kn_model.predict(df_submit[features])

    df_submit['test_id'] = df_submit['test_id'].astype(int)

    df_submit['price'] = predictions

    df_submit.to_csv('submit.csv', columns=["test_id", "price"], index=False)