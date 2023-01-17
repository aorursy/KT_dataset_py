import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from math import sqrt
from itertools import product
from sklearn.metrics import mean_squared_error
# loading dataset
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales_train['date_block_num'].unique():
    cur_shops = sales_train[sales_train['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales_train[sales_train['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Aggregations
sales_train['item_cnt_day'] = sales_train['item_cnt_day'].clip(0,20) #removing outliers
groups = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])
trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
trainset = trainset.rename(columns = {'item_cnt_day' : 'item_cnt_month'})
trainset['item_cnt_month'] = trainset['item_cnt_month'].clip(0,20)

trainset = pd.merge(grid,trainset,how='left',on=index_cols)
trainset.item_cnt_month = trainset.item_cnt_month.fillna(0)

# Get category id
trainset = pd.merge(trainset, items[['item_id', 'item_category_id']], on = 'item_id')
trainset.to_csv('trainset_with_grid.csv')

trainset.head()
# Extract features and target we want
baseline_features = ['shop_id', 'item_id', 'item_category_id', 'date_block_num', 'item_cnt_month']
train = trainset[baseline_features]
# Remove pandas index column
train = train.set_index('shop_id')
train.item_cnt_month = train.item_cnt_month.astype(int)
train['item_cnt_month'] = train.item_cnt_month.fillna(0).clip(0,20)
# Save train set to file
train.to_csv('train.csv')
train = train.reset_index()
train

# taking last month from train for validation
Training_Data = train[train['date_block_num'] < 33]
validation = train[(train['date_block_num'] == 33)]
y_train = Training_Data['item_cnt_month']
x_train = Training_Data.drop('item_cnt_month', axis = 1)
y_validation =  validation['item_cnt_month']
x_validation = validation.drop('item_cnt_month', axis = 1)
print (x_train.shape, x_validation.shape, y_train.shape,y_validation.shape )
# LR as benchmark
import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train, y_train)
pred = reg.predict(x_validation)
res = sqrt(mean_squared_error(pred,y_validation))
print(res)
#setting test set as the training dataframe for predicting
test_df = pd.DataFrame(test, columns = ['shop_id', 'item_id'])
# Make test_dataset pandas data frame, add category id and date block num, then convert back to numpy array and predict
X_test = pd.merge(test_df, items, on = ['item_id'])[['shop_id','item_id','item_category_id']]
X_test['date_block_num'] = 34
X_test
test_pred = reg.predict(X_test)
submission1_dataframe = pd.DataFrame(test_pred)
submission1_dataframe = submission1_dataframe.reset_index()
submission1_dataframe.columns = ["ID", "item_cnt_month"]
submission1_dataframe = submission1_dataframe.set_index('ID')
submission1_dataframe
submission1_dataframe.to_csv("submission_LR.csv")
# score in kaggle was 1.21242

#setting general callback function
def set_callbacks(description='run1',patience=5,tb_base_logdir='./logs/'):
    from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
    cp = ModelCheckpoint('best_model_weights_{}.h5'.format(description),save_best_only=True)
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=5)
    return [rlop, es, cp]
# # DL without embedding
# from keras.models import Sequential
# from keras.layers import Dense
# # create model
# model = Sequential()
# model.add(Dense(1000, input_dim=4, kernel_initializer='normal', activation='relu'))
# model.add(Dense(500, kernel_initializer='normal', activation='relu'))
# model.add(Dense(100, kernel_initializer='normal', activation='relu'))
# model.add(Dense(1, kernel_initializer='normal'))
# # Compile model
# model.compile(loss='mean_squared_error', optimizer='adam')
# hist = model.fit(x_train.values, y_train.values, epochs = 10 , validation_data = (x_validation.values,y_validation.values),
#                  batch_size = 256, callbacks = set_callbacks())
# test_pred = model.predict(X_test)
# submission1_dataframe = pd.DataFrame(test_pred)
# submission1_dataframe = submission1_dataframe.reset_index()
# submission1_dataframe.columns = ["ID", "item_cnt_month"]
# submission1_dataframe = submission1_dataframe.set_index('ID')
# submission1_dataframe
# submission1_dataframe.to_csv("submission_DL.csv")
# # score in kaggle was 1.21756

# DL model with embedding, based 
#input length for embedding layer
items_count = train['item_id'].unique()
shops_count = train['shop_id'].unique()
item_category_count = train['item_category_id'].unique()
item_month_count = train['item_cnt_month'].unique()
# DL model with embedding, based 
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Flatten, Input, Dropout, BatchNormalization, Concatenate, Add, add, concatenate
from keras.utils import plot_model
from keras.regularizers import l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt
%matplotlib inline

item_inp = Input(shape=(1,),dtype='int64')
shops_inp = Input(shape=(1,),dtype='int64')
item_cat_inp = Input(shape=(1,),dtype='int64')
month_inp = Input(shape=(1,),dtype='int64')

item_emb = Embedding(len(items_count),50,input_length=1, embeddings_regularizer=l2(1e-6))(item_inp)
shops_emb = Embedding(len(shops_count),7,input_length=1, embeddings_regularizer=l2(1e-6))(shops_inp)
item_cat_emb = Embedding(len(item_category_count),8,input_length=1, embeddings_regularizer=l2(1e-6))(item_cat_inp)
month_emb = Embedding(34,5,input_length=1, embeddings_regularizer=l2(1e-6))(month_inp)

item_bias = Embedding(len(items_count), 1, input_length=1)(item_inp)
shop_bias = Embedding(len(shops_count), 1, input_length=1)(shops_inp)
month_bias = Embedding(len(item_month_count), 1, input_length=1)(month_inp)
item_cat_bias = Embedding(len(item_category_count), 1, input_length=1)(item_cat_inp)

a = add([item_emb, item_bias])
b = add([shops_emb, shop_bias])
c = add([item_cat_emb, item_cat_bias])
d = add([month_emb, month_bias])

x = concatenate([a, b, c, d],name='concatenate')
x = Flatten(name='Flatten')(x)
x = Dense(1000,activation='relu')(x)
x = Dense(500,activation='relu')(x)
x = Dense(100,activation='relu')(x)
x = Dense(1,activation='relu',kernel_initializer='normal')(x)
model_2 = Model([item_inp,shops_inp,item_cat_inp,month_inp],x)
model_2.compile(loss = 'mean_squared_error',optimizer='adam')
# plot_model(model_2, show_shapes=True)
model_2.summary()
train
hist = model_2.fit([x_train[x].values for x in x_train.columns],y_train.values,
          validation_data = ([x_validation[x].values for x in x_validation.columns],y_validation.values),
          batch_size = 256, epochs=30, callbacks = set_callbacks())
X_test
submission1 = model_2.predict([X_test[x].values for x in X_test.columns])
submission1_dataframe = pd.DataFrame(submission1)
submission1_dataframe = submission1_dataframe.reset_index()
submission1_dataframe.columns = ["ID", "item_cnt_month"]
submission1_dataframe = submission1_dataframe.set_index('ID')
submission1_dataframe
submission1_dataframe.to_csv("submission_DL_EMBEDDING_25_5.csv")

#### add season and holiday 
import holidays
ru_holiday = holidays.RU(years = [2013,2014,2015])
holiday_env = set()

for holiday in ru_holiday:
    holiday_env.add((holiday.month - 1, holiday.year))
    holiday_env.add((holiday.month, holiday.year))


added_data = train.copy()

added_data.insert(len(added_data.columns),'winter',allow_duplicates = True, value = 0)
added_data.insert(len(added_data.columns),'fall',allow_duplicates = True, value = 0)
added_data.insert(len(added_data.columns),'spring',allow_duplicates = True, value = 0)
added_data.insert(len(added_data.columns),'summer',allow_duplicates = True, value = 0)
added_data.insert (len(added_data.columns),'holiday',allow_duplicates = True, value = 0)


for idx,row in added_data.iterrows():
    month = int(row['date_block_num'] + 1 % 12)
    if (month >= 3 and month < 6):
        added_data['spring'][idx] = 1
    elif (month >= 6 and month < 9):
        added_data['summer'][idx] = 1
    elif (month >= 9 and month < 11):
        added_data['fall'][idx] = 1
    else:
        added_data['winter'][idx] = 1
    year = int(row['date_block_num'] / 12) + 2013
    if (month, year) in holiday_env:
        added_data['holiday'][idx] = 1

added_data.to_csv("./date_and_holiday_ass2_q3.csv", index=True)
added_data
added_data = added_data.fillna(0)

x_train = added_data[added_data.date_block_num <= 32]
x_val = added_data[added_data.date_block_num == 33]

y_train = x_train['item_cnt_month']
x_train = x_train.drop('item_cnt_month', axis = 1)

y_val = x_val['item_cnt_month']
x_val = x_val.drop('item_cnt_month',axis = 1)


x_train
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Flatten, Input, Dropout, BatchNormalization, Concatenate, Add, add, concatenate
from keras.utils import plot_model
from keras.regularizers import l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt
%matplotlib inline
# from keras.layers import *
from keras.callbacks import *
# from keras.optimizers import *
# from keras.utils import to_categorical

train = added_data


def set_callbacks(description='run1',patience=8,tb_base_logdir='./logs/'):
    cp = ModelCheckpoint('best_model_weights_{}.h5'.format(description),save_best_only=True)
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=5)
    return [rlop, es, cp]

item_max = train['item_id'].max() + 1
shop_max = train['shop_id'].max() + 1
item_cat_max = train['item_category_id'].max() + 1


winter_inp = Input(shape=(1,),dtype='int64')
fall_inp = Input(shape=(1,),dtype='int64')
spring_inp = Input(shape=(1,),dtype='int64')
summer_inp = Input(shape=(1,),dtype='int64')
holiday_inp = Input(shape=(1,),dtype='int64')



winter_emb = Embedding(2,1,input_length=1, embeddings_regularizer=l2(1e-6), embeddings_initializer='Ones')(winter_inp)
fall_emb = Embedding(2,1,input_length=1, embeddings_regularizer=l2(1e-6),embeddings_initializer='Ones')(fall_inp)
spring_emb = Embedding(2,1,input_length=1, embeddings_regularizer=l2(1e-6),embeddings_initializer='Ones')(spring_inp)
summer_emb = Embedding(2,1,input_length=1, embeddings_regularizer=l2(1e-6),embeddings_initializer='Ones')(summer_inp)
holiday_emb = Embedding(2,1,input_length=1, embeddings_regularizer=l2(1e-6),embeddings_initializer='Ones')(holiday_inp)


item_inp = Input(shape=(1,),dtype='int64')
shops_inp = Input(shape=(1,),dtype='int64')
item_cat_inp = Input(shape=(1,),dtype='int64')
month_inp = Input(shape=(1,),dtype='int64')

item_emb = Embedding(item_max, 50,input_length=1, embeddings_regularizer=l2(1e-6))(item_inp)
shops_emb = Embedding(shop_max,8,input_length=1, embeddings_regularizer=l2(1e-6))(shops_inp)
item_cat_emb = Embedding(item_cat_max,10,input_length=1, embeddings_regularizer=l2(1e-6))(item_cat_inp)
month_emb = Embedding(34,6,input_length=1, embeddings_regularizer=l2(1e-6))(month_inp)


item_bias = Embedding(item_max, 1, input_length=1)(item_inp)
shop_bias = Embedding(shop_max, 1, input_length=1)(shops_inp)
item_cat_bias = Embedding(item_cat_max, 1, input_length=1)(item_cat_inp)
month_bias = Embedding(34, 1, input_length=1)(month_inp)


a = add([item_emb, item_bias])
b = add([shops_emb, shop_bias])
d = add([item_cat_emb, item_cat_bias])
e = add([month_emb, month_bias])


# x = concatenate([a,b,d,e,winter_inp, fall_inp, spring_inp, summer_inp, holiday_inp])
x = concatenate([a,b,d,e,winter_emb, fall_emb, spring_emb, summer_emb, holiday_emb])
x = Flatten()(x)
x = Dense(1000,activation='relu')(x)
x = Dense(500,activation='relu')(x)
x = Dense(100,activation='relu')(x)
x = Dense(1,activation='relu',kernel_initializer='normal')(x)

model_2 = Model([shops_inp, item_inp ,month_inp, item_cat_inp, winter_inp, fall_inp, spring_inp, summer_inp, holiday_inp],x)
model_2.compile(loss = 'mean_squared_error',optimizer='adam')

model_2.summary()
hist = model_2.fit([x for x in input_cols],y_train.values,
          validation_data = ([x for x in val_cols],y_val.values), validation_steps = 930,
          steps_per_epoch = 41701, epochs=30, callbacks = set_callbacks())
#Feature extraction
layer_name = 'Flatten'
intermediate_layer_model = Model(inputs=model_2.input,outputs=model_2.get_layer(layer_name).output)
Embedded_Features = intermediate_layer_model.predict([x_train[x].values for x in x_train.columns])
#Feature extraction
Embedded_Features_validation = intermediate_layer_model.predict([x_validation[x].values for x in x_validation.columns])
Embedded_Features.shape
#Feature extraction
Embedded_Features_test = intermediate_layer_model.predict([X_test[x].values for x in X_test.columns])
import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(Embedded_Features, y_train)
pred = reg.predict(Embedded_Features_validation)
mse = mean_squared_error(pred,y_validation)
rmse = sqrt(mean_squared_error(pred,y_validation))
print(mse)

print(rmse)

pred = reg.predict(Embedded_Features_test)
submission1_dataframe = pd.DataFrame(pred)
submission1_dataframe = submission1_dataframe.reset_index()
submission1_dataframe.columns = ["ID", "item_cnt_month"]
submission1_dataframe = submission1_dataframe.set_index('ID')
submission1_dataframe
submission1_dataframe.to_csv("submission_Linear_Regressor.csv")