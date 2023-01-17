N_EPOCHS = 75

EARLY_STOPPING = 5

BATCH_SIZE = 512

FOLD_NUMBER = 0



def reduce_mem_usage(df, verbose=True, downcast_float=False):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if downcast_float:

                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                        df[col] = df[col].astype(np.float16)

                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                        df[col] = df[col].astype(np.float32)

                    else:

                        df[col] = df[col].astype(np.float64)    

                

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
class PreprocessingUnit(object):

    def __init__(self, categoricals, conts, replace_nans, target_variable):

        self.categoricals = categoricals

        self.conts = conts

        self.replace_nans = replace_nans

        self.target_variable = target_variable

        

        self.scaling_stats = {}

        self.col_label_codes = {}

        self.predictors = None

        

    def train_preprocessing(self, data):

        for col in self.replace_nans:

            data.loc[data[col] == -1, col] = data.loc[data[col] != -1, col].median()

        

        y = data[self.target_variable]

        data = data.drop(columns=self.target_variable)

        

        self.predictors = list(data.columns)

        

        for col in self.conts:

            self.scaling_stats[col] = {'mean':data[col].mean(), 'std':data[col].std()}

            data[col] = (data[col] - self.scaling_stats[col]['mean']) / self.scaling_stats[col]['std']

            

        print(data[self.conts].isna().sum().sum())

        print('Scaling completed!')

        

        cat_columns = []

        for col in self.categoricals:

            self.col_label_codes[col] = {k:v for v, k in enumerate(data[col].unique())}

            cat_columns.append(data[col].map(self.col_label_codes[col]).values)

            data = data.drop(columns=col)

            gc.collect()

        

        print('Labeling completed!')

        

        gc.collect()

        

        return [data.values] + cat_columns, y.values

    

    def test_preprocessing(self, data, is_val=False):

        for col in self.replace_nans:

            data.loc[data[col] == -1, col] = data.loc[data[col] != -1, col].median()

        

        if is_val:

            y = data[self.target_variable]

            data = data.drop(columns=self.target_variable)

            

        data = data[self.predictors]

        gc.collect()

        

        for col in self.conts:

            data[col] = (data[col] - self.scaling_stats[col]['mean']) / self.scaling_stats[col]['std']

            

        cat_columns = []

        for col in self.categoricals:

            cat_columns.append(data[col].map(self.col_label_codes[col]).values)

            data = data.drop(columns=col)

            gc.collect()

        

        gc.collect()

        if is_val:

            return [data.values] + cat_columns, y.values

        else:

            return [data.values] + cat_columns
from tqdm import tqdm



import keras.backend as K



from keras.layers import Input, Dense, Dropout, Embedding, Concatenate, Lambda

from keras.models import Model

from keras.optimizers import Adam, Nadam, Adamax

from keras import callbacks



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder





es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=EARLY_STOPPING, verbose=False, mode='auto', restore_best_weights=True)

rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, mode='auto', verbose=False)



def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true)))



def create_model(inp_dim):

    inps = Input(shape=(inp_dim,))

    build_inp = Input(shape=(1,))

    meter_inp = Input(shape=(1,))

    site_inp = Input(shape=(1,))

    prime_inp = Input(shape=(1,))

    dow_inp = Input(shape=(1,))

    hod_inp = Input(shape=(1,))

    

    build_embed = Embedding(1449, 38)(build_inp)

    build_embed = Lambda(lambda x: K.squeeze(x, 1))(build_embed)

    

    meter_embed = Embedding(4, 2)(meter_inp)

    meter_embed = Lambda(lambda x: K.squeeze(x, 1))(meter_embed)

    

    site_embed = Embedding(16, 4)(site_inp)

    site_embed = Lambda(lambda x: K.squeeze(x, 1))(site_embed)

    

    prime_embed = Embedding(16, 4)(site_inp)

    prime_embed = Lambda(lambda x: K.squeeze(x, 1))(prime_embed)

    

    dow_embed = Embedding(7, 3)(dow_inp)

    dow_embed = Lambda(lambda x: K.squeeze(x, 1))(dow_embed)

    

    hod_embed = Embedding(24, 5)(hod_inp)

    hod_embed = Lambda(lambda x: K.squeeze(x, 1))(hod_embed)

        

    x = Concatenate(axis=-1)([inps,build_embed,meter_embed,site_embed,prime_embed, dow_embed, hod_embed])

    

    x = Dense(256, activation='elu')(x)

    x = Dropout(0.2)(x)

    x = Dense(128, activation='elu')(x)

    x = Dropout(0.1)(x)

    x = Dense(1, activation='softplus')(x)

    model = Model(inputs=[inps,build_inp,meter_inp,site_inp,prime_inp,dow_inp,hod_inp], outputs=x)

    model.compile(

        optimizer=Adamax(lr=1e-3),

        loss=root_mean_squared_error

    )

    return model

        

        
import pandas as pd

import numpy as np

import gc



from os import path



%matplotlib inline
cat_columns = ["building_id", "meter", "site_id", "primary_use", "tm_day_of_week", "tm_hour_of_day"]

target_column = "meter_reading"

fold_col = 'k_folds'

cont_columns = [

 'air_temperature', 'air_temperature_max_lag24', 'air_temperature_mean_lag24', 'air_temperature_median_lag24',

 'air_temperature_min_lag24', 'cloud_coverage', 'dew_temperature', 'dew_temperature_max_lag24',

 'dew_temperature_mean_lag24', 'dew_temperature_median_lag24', 'dew_temperature_min_lag24',

 'floor_count', 'max_at', 'mean_dt', 'min_at', 'min_dt', 'precip_depth_1_hr', 'sea_level_pressure',

 'square_feet', 'wind_direction', 'wind_speed', 'year_built'

]

required_columns = cat_columns + cont_columns
pp = PreprocessingUnit(categoricals=cat_columns, 

                       conts=cont_columns, 

                       replace_nans=['year_built', 'floor_count', 'precip_depth_1_hr'], 

                       target_variable=target_column)
X_train = reduce_mem_usage(pd.read_parquet('/kaggle/input/baseline-preprocessing-leaks-train-fe/X_train.parquet.gzip')[required_columns + [target_column,fold_col]])



X_train, X_val = X_train[X_train['k_folds'] != FOLD_NUMBER].reset_index(drop=True), X_train[X_train['k_folds'] == FOLD_NUMBER].reset_index(drop=True)



X_train = X_train.drop(columns='k_folds')

X_val = X_val.drop(columns='k_folds')

print(X_train.columns)

gc.collect()



X_train, y_train = pp.train_preprocessing(X_train)

gc.collect()

X_val, y_val = pp.test_preprocessing(X_val, is_val=True)
gc.collect()
neural_net = create_model(X_train[0].shape[1])
history = neural_net.fit(

            X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose=True, callbacks=[es, rlr]

        )
pd.DataFrame(history.history)
del X_train

del y_train

del X_val

del y_val



gc.collect()
del history



gc.collect()
X_test = reduce_mem_usage(pd.read_parquet('/kaggle/input/baseline-preprocessing-leaks-train-fe/X_test.parquet.gzip')[required_columns], downcast_float=True)

X_test = pp.test_preprocessing(X_test, is_val=False)
gc.collect()
prediction = neural_net.predict(X_test, batch_size=BATCH_SIZE, verbose=True)
prediction = prediction.flatten()
np.save('prediction.npy', prediction)