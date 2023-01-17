import numpy as np
import pandas as pd
from sklearn import preprocessing

import gc
from keras.models import Model
from keras.layers import Input, Dense

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Input, BatchNormalization, Dense, Reshape, Lambda, Dropout
from keras import metrics
#from . import backend as K
from keras import backend as K



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv('../input/train_2016_v2.csv', nrows=13200) 
prop = pd.read_csv('../input/properties_2016.csv', nrows=13200) 
sample = pd.read_csv('../input/sample_submission.csv')







prop['propertycountylandusecode'] = prop['propertycountylandusecode'].apply(lambda x: str(x))
encoder = preprocessing.LabelEncoder()
encoder.fit(prop['propertycountylandusecode'])
prop['propertycountylandusecode'] = encoder.transform(prop['propertycountylandusecode'])

prop['propertyzoningdesc'] = prop['propertyzoningdesc'].apply(lambda x: str(x))
encoder2 = preprocessing.LabelEncoder()
encoder2.fit(prop['propertyzoningdesc'])
prop['propertyzoningdesc'] = encoder2.transform(prop['propertyzoningdesc'])



# Discard all non-numeric data
prop = prop.select_dtypes([np.number])
train = train.select_dtypes([np.number])
sample = sample.select_dtypes([np.number])

gc.collect()


x_train = prop.drop(['parcelid'], axis=1)

gc.collect()



train_columns = x_train.columns
temp = pd.merge(left=train, right=prop, on=('parcelid'), how='outer')
temp = temp.fillna(0)
x_train = temp.drop(['parcelid', 'logerror'], axis=1).values
y_train = temp['logerror'].values
gc.collect()


scaler = preprocessing.StandardScaler()
# x_train.apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
x_train = scaler.fit_transform(x_train)

# Normalize (across the whole dataframe cos we dun care)
mean_x = x_train.mean().astype(np.float32)
std_x = x_train.std().astype(np.float32)

mean_y = y_train.mean().astype(np.float32)
std_y = y_train.std().astype(np.float32)

def normalize(x):
    return (x-mean_x)/std_x

def normalize_y(y):
    return (y-mean_y)/std_y

def de_normalize_y(y):


    return (y*std_y) + mean_y

y_train = normalize(y_train)




# Build a simple model
model = Sequential([
    #Lambda(normalize,input_shape=(52, )),
	Dense(60,input_shape=(54, )),
    BatchNormalization(),
    Dropout(0.08),
	Dense(160, activation='relu'),
	BatchNormalization(),
    Dropout(0.38),
    Dense(20, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])




model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(x_train, y_train, batch_size=24, epochs=15)



model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, batch_size=32, epochs=25)

# Prepare the submission data
sample['parcelid'] = sample['ParcelId']
del sample['ParcelId']

df_test = pd.merge(sample, prop, on='parcelid', how='left')
df_test = df_test.fillna(0)

x_test = df_test[train_columns]
#predict(self, x, batch_size=None, verbose=0, steps=None)
p_test = model.predict(x_test.values)
p_test = de_normalize_y(p_test)
p_test
# evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
q_test=model.evaluate(x_train,y_train)
#q_test= de_normalize(q_test)


q_test
q_test= de_normalize_y(q_test)
q_test
#train_on_batch(self, x, y, sample_weight=None, class_weight=None)
model.train_on_batch(x_train,y_train)
#test_on_batch(self, x, y, sample_weight=None)
model.test_on_batch(x_train,y_train, sample_weight=None)

#predict_on_batch(self, x)
model.predict_on_batch(x_train)
#model.evaluate_on_batch(x_train,y_train)
###fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

#model.fit_generator(x_train, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
#evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
#model.evaluate_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
#predict_generator(self, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
#model.predict_generator(x_train, generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
#get_layer(self, name=None, index=None)
#lay=model.get_layer(x_train.all())
print(lay)
sub = pd.read_csv('../input/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('output2.csv', index=False, float_format='%.5f')

sub.head()








