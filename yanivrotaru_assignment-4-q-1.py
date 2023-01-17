# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model

from keras.layers.pooling import MaxPooling1D
from keras.layers import Input, Dense, Flatten,Lambda,Conv1D,LSTM,Subtract,GlobalMaxPooling1D,Activation

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
%matplotlib inline

def set_callbacks( name = 'best_model_weights',patience=10):
#     from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
    cp = ModelCheckpoint(name +'.h5',save_best_only=True)
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=5)
    return [rlop, es, cp]


%time
# attributes = pd.read_csv('attributes.csv')
product_descriptions = pd.read_csv('../input/home-depot-product-search-relevance/product_descriptions.csv.zip')
test = pd.read_csv('../input/home-depot-product-search-relevance/test.csv.zip',encoding='latin-1')
train = pd.read_csv('../input/home-depot-product-search-relevance/train.csv.zip',encoding='latin-1')
solution = pd.read_csv('../input/solution/solution.csv')
# sample_submission = pd.read_csv('sample_submission.csv')
train_df = pd.merge(train, product_descriptions, on='product_uid')
test_df = pd.merge(test, product_descriptions, on='product_uid')

test_df




solution
test_df = test_df[solution["Usage"]!='Ignored']
test_df.insert(4, 'relevance',  solution.relevance)
train_df

test_df
train_df["search_term"] = train_df["search_term"].str.lower()
train_df["product_description"] = train_df["product_description"].str.lower()
test_df["search_term"] = test_df["search_term"].str.lower()
test_df["product_description"] = test_df["product_description"].str.lower()
train_df.head()
train_df.search_term = [list(i) for i in train_df.search_term]
train_df.product_description = [list(i) for i in train_df.product_description]
test_df.search_term = [list(i) for i in test_df.search_term]
test_df.product_description = [list(i) for i in test_df.product_description]
train_df
test_df
test_set = test_df[['search_term', 'product_description']]
train_set = train_df[['search_term', 'product_description',]]
y =  train_df['relevance']
y_test = test_df['relevance'].values
# %%time
# # PreProcess the train set - char to int, this action takes time
# for index, row in train_set.iterrows():
#     train_set.search_term[index] = np.array([ord(i) for i in row.search_term])
#     train_set.product_description[index] = np.array([ord(i) for i in row.product_description])
# train_set.to_pickle("train_Char_to_int")
train_set = pd.read_pickle("../input/train-char-to-int/train_Char_to_int")
train_set
# %%time
# # PreProcess the test set - char to int, this action takes time
# for index, row in test_set.iterrows():
#     test_set.search_term[index] = np.array([ord(i) for i in row.search_term])
#     test_set.product_description[index] = np.array([ord(i) for i in row.product_description])
# test_set.to_pickle("test_Char_to_int")
test_set = pd.read_pickle("../input/train-char-to-int/test_Char_to_int")
test_set
maxlen_search = train_set.search_term.apply(len).max()
# 10% safe margin for padding
search_safe_margin = int(maxlen_search*0.1)
maxlen_desc = train_set.product_description.apply(len).max()
desc_margin_safe_margin = int(maxlen_desc*0.1)
print('maximum length of search term in train set is:' ,maxlen_search, '\nsafe margin for padding is:',search_safe_margin,
      '\npadded size of search term is:',maxlen_search+ search_safe_margin,
      '\n\nmaximum length of product description in train set is',maxlen_desc,
      '\nsafe margin for padding is:',desc_margin_safe_margin , '\npadded size of product description is:',maxlen_desc+ desc_margin_safe_margin)
maxlen_search_test = test_set.search_term.apply(len).max()
maxlen_desc_test = test_set.product_description.apply(len).max()

print('maximum length of search term in test set is:' ,maxlen_search_test,
      '\nmaximum length of product description in test set is',maxlen_desc_test)
X_train, X_val, y_train, y_val = train_test_split(train_set,y.values, test_size=0.2, random_state=42)
X_train
X_test = test_set
X_test
search_term_train = np.array(pad_sequences(X_train.search_term, maxlen=maxlen_search))
# product_description_train = np.array(pad_sequences(X_train.product_description, maxlen=maxlen_desc))
product_description_train = np.array(pad_sequences(X_train.product_description, maxlen=1000)) #due to long runtime

search_term_val = np.array(pad_sequences(X_val.search_term, maxlen=maxlen_search))
# product_description_val = np.array(pad_sequences(X_val.product_description, maxlen=maxlen_desc))
product_description_val = np.array(pad_sequences(X_val.product_description, maxlen=1000)) #due to long run time

search_term_test = np.array(pad_sequences(X_test.search_term, maxlen=maxlen_search))
# product_description_test = np.array(pad_sequences(X_test.product_description, maxlen=maxlen_desc))
product_description_test = np.array(pad_sequences(X_test.product_description, maxlen=1000)) #due to long run time

print('search_term shape is',search_term_train.shape,'\nproduct description shape is',product_description_train.shape, )
print('search_term shape is',search_term_val.shape,'\nproduct description shape is',product_description_val.shape, )
print('search_term shape is',search_term_test.shape,'\nproduct description shape is',product_description_test.shape, )

def euclidean_distance(vecs):
    return K.sqrt(K.sum(vecs, axis=1, keepdims=True))

def subs_square(vecs):
    x, y = vecs
    return K.square(x - y)

#with LSTM
def siamese_model(shape):
    inp = Input(shape)
    x = (Conv1D(64, (10),activation='relu'))(inp)
    x = GlobalMaxPooling1D()(x)
    x = Activation('relu')(x)
    x = Dense(128,activation = 'relu')(x)
    return Model(inputs=inp, outputs=x)
siamese_model((128,1)).summary()
inp_product = Input((product_description_train.shape[1],1))
inp_search = Input((search_term_train.shape[1],1))
lstm_product = LSTM(128)(inp_product)
lstm_search = LSTM(128)(inp_search)
reshape_layer = Lambda(lambda tensor: tensor[...,np.newaxis],name="reshape_lstm_output")
reshape_product = reshape_layer(lstm_product)
reshape_search = reshape_layer(lstm_search)
model = siamese_model((128,1))
encoded_l = model(reshape_search)
encoded_r = model(reshape_product)
FE = Lambda(subs_square)([encoded_l, encoded_r])
x = Lambda(euclidean_distance)(FE)
siamese_net = Model(inputs=[inp_product,inp_search],outputs=x)
siamese_net.summary()
siamese_net.compile(optimizer='adam', loss='mse',metrics=['mae'])
plot_model(siamese_net)
%%time
hist = siamese_net.fit([np.expand_dims(product_description_train,2),np.expand_dims(search_term_train,2)],y_train,
                validation_data = ([np.expand_dims(product_description_val,2),np.expand_dims(search_term_val,2)],y_val),
                epochs = 30,batch_size=256,callbacks = set_callbacks(name = 'siamese_model_weights'))
def print_loss_psnr(hist,title_name):
    plt.figure()
    plt.plot(hist.history['loss'], 'r', hist.history['val_loss'], 'b')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title(title_name + ' loss')
    plt.legend(['loss','val_loss'])
print_loss_psnr(hist,title_name = 'Siamese model loss')
test_pred = siamese_net.predict([np.expand_dims(product_description_test,2),np.expand_dims(search_term_test,2)])
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
Siamese_mae_test = mean_absolute_error(y_test, test_pred)
Siamese_rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))

print('Siamese model test mae is:', Siamese_mae_test)
print('Siamese model test rmse is:', Siamese_rmse_test)
%time
product_descriptions = pd.read_csv('../input/home-depot-product-search-relevance/product_descriptions.csv.zip')
test = pd.read_csv('../input/home-depot-product-search-relevance/test.csv.zip',encoding='latin-1')
train = pd.read_csv('../input/home-depot-product-search-relevance/train.csv.zip',encoding='latin-1')
solution = pd.read_csv('../input/solution/solution.csv')

train_df = pd.merge(train, product_descriptions, on='product_uid')
test_df = pd.merge(test, product_descriptions, on='product_uid')
train_df["search_term"] = train_df["search_term"].str.lower()
train_df["product_description"] = train_df["product_description"].str.lower()
test_df["search_term"] = test_df["search_term"].str.lower()
test_df["product_description"] = test_df["product_description"].str.lower()
test_df = test_df[solution["Usage"]!='Ignored']
test_df.insert(4, 'relevance',  solution.relevance)
df_train = pd.DataFrame(columns=['search_term', 'product_description','relevance'], data=train_df[['search_term','product_description','relevance']].values)
df_test = pd.DataFrame(columns=['search_term', 'product_description','relevance'], data=test_df[['search_term','product_description','relevance']].values)
df_train
df_test
%%time
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 4000,analyzer='char')
X1_train=cv.fit_transform(df_train.product_description).toarray()
X2_train=cv.transform(df_train.search_term).toarray()

X1_test=cv.fit_transform(df_test.product_description).toarray()
X2_test=cv.transform(df_test.search_term).toarray()
y = df_train.relevance.values
X = np.concatenate((X1_train,X2_train),axis = 1)
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)
X_test =  np.concatenate((X1_test,X2_test),axis = 1)
y_test = df_test.relevance.values
%%time
from xgboost import XGBRegressor
XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
def scores(model,name,X_train,X_val,X_test,y_train,y_val,y_test):
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(str(name) + ' train MAE is:', train_mae)
    print(str(name) + ' train RMSE is:', train_rmse)

    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(str(name) + ' validation MAE is:', val_mae)
    print(str(name) + ' validation RMSE is:', val_rmse)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print(str(name) + ' test MAE is:', test_mae)
    print(str(name) + ' test RMSE is:', test_rmse)

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
scores(XGB_model,'XGB_regressor',X_train,X_val,X_test,y_train,y_val,y_test)
%time
FE_model = Model(inputs=[inp_product,inp_search],outputs=FE)
FE_model.summary()

import seaborn as sns
FE_train = FE_model.predict([np.expand_dims(product_description_train,2),np.expand_dims(search_term_train,2)]) #Feature extractor
FE_val = FE_model.predict([np.expand_dims(product_description_val,2),np.expand_dims(search_term_val,2)]) #Feature extractor
FE_test = FE_model.predict([np.expand_dims(product_description_test,2),np.expand_dims(search_term_test,2)]) #Feature extractor
plot_model(FE_model)
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
%%time
from xgboost import XGBRegressor
XGB_model = XGBRegressor()
XGB_model.fit(FE_train, y_train)
scores(XGB_model,'XGB_regressor',FE_train,FE_val,FE_test,y_train,y_val,y_test)
%%time
sgd = SGDRegressor()
sgd.fit(FE_train, y_train)

scores(sgd,'SGD',FE_train,FE_val,FE_test,y_train,y_val,y_test)

%%time
xgboost = GradientBoostingRegressor(random_state=42)
xgboost.fit(FE_train, y_train)


scores(xgboost,'XGB_Boost',FE_train,FE_val,FE_test,y_train,y_val,y_test)

