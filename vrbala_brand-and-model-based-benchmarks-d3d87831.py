import pandas as pd

import numpy as np

from scipy.sparse import csr_matrix

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

import sklearn

import os

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix, hstack
# read data

data_dir = '../input'

phones = pd.read_csv(os.path.join(data_dir, 'phone_brand_device_model.csv'), encoding='utf-8')

app_labels = pd.read_csv(os.path.join(data_dir, 'app_labels.csv'), index_col=0)

events = pd.read_csv(os.path.join(data_dir, 'events.csv'), parse_dates=['timestamp'])

gatest = pd.read_csv(os.path.join(data_dir, 'gender_age_test.csv'))

gatrain = pd.read_csv(os.path.join(data_dir, 'gender_age_train.csv'))

label_categories = pd.read_csv(os.path.join(data_dir, 'label_categories.csv'))

app_events = pd.read_csv(os.path.join(data_dir, 'app_events.csv'), dtype={'is_active': np.bool})
# data cleanup

phones = phones.drop_duplicates(["device_id"])

app_events = app_events[app_events.is_active == True]
print("#phones: {}".format(len(phones.device_id.unique())))

print("#events: {}".format(len(events.event_id.unique())))

print("#app_events: {}".format(len(app_events)))

#print("#app_labels: {}".format(len(app_labels.app_id.unique())))

print("#gatrain: {}".format(len(gatrain)))

print("#gatest: {}".format(len(gatest)))
# no duplicate device ids in phones now

len(phones) - phones.device_id.value_counts().sum()
# no duplicate app/event combo.

app_events.groupby(["event_id", "app_id"], as_index=False).count().sort_values(by="is_active", ascending=False)
# no duplicate rows per device in training set

gatrain.groupby(["device_id"]).count().group.unique()
def oneHotEncode(data):

    encoder = LabelEncoder()

    encoder.fit(data)

    return encoder.transform(data)
# start setting up the sparse matrix of test data

gatrain["row_num"] = np.arange(len(gatrain))

gatrain["group"] = oneHotEncode(gatrain.group)

# output labels

y = csr_matrix((np.ones(len(gatrain)), (gatrain.row_num.tolist(), gatrain.group.tolist())))

print(y.shape)
# build sparse matrices for brand and model

phones["brand_model"] = np.add(phones.phone_brand, phones.device_model)

phones["brand"] = oneHotEncode(phones.phone_brand)

phones["model"] = oneHotEncode(phones.brand_model)

_tmpData = gatrain.merge(phones, on="device_id")

X_brand = csr_matrix((np.ones(len(gatrain)), (gatrain.row_num.tolist(), _tmpData.brand.tolist())))

X_model = csr_matrix((np.ones(len(gatrain)), (gatrain.row_num.tolist(), _tmpData.model.tolist())))

del(_tmpData)

print(X_brand.shape)

print(X_model.shape)
# build sparse matrix for (device, hour_used) -> #times used

events["hour"] = events.timestamp.dt.hour

_tmpData = gatrain.merge(events.groupby(["device_id", "hour"], as_index=False).count(),

                        on="device_id", how="left")[["row_num", "device_id", "hour", "event_id"]] # any column instead of event_id will work equally well!

_tmpData["hour"] = oneHotEncode(_tmpData.hour.fillna(25)) # I know, invalid hour. But that means, we don't have usage info for this device.

_tmpData["counts"] = _tmpData.event_id.fillna(0)

_tmpData = _tmpData.drop(["event_id"], axis=1)

X_hour_used = csr_matrix((_tmpData.counts, (_tmpData.row_num.tolist(), _tmpData.hour.tolist())))

del(_tmpData)

print(X_hour_used.shape)
# build sparse matrix of label categories

app_labels["id"] = oneHotEncode(app_labels.label_id)

_tmpData = events.merge(app_events, on="event_id")[["device_id", "app_id"]]

_tmpData.set_index("device_id", inplace=True)

_tmpData = _tmpData.merge(app_labels, left_on="app_id", right_index=True)

_tmpData = _tmpData.drop(["app_id", "label_id"], axis=1)

_tmpData = gatrain.merge(_tmpData, left_on="device_id", right_index=True)
X_categories = csr_matrix((np.ones(len(_tmpData)), (_tmpData.row_num.tolist(), _tmpData.id.tolist())))

print(X_categories.shape)
X = hstack([X_brand, X_model, X_hour_used, X_categories])

print(X.shape)
print(X)
# each app has more than one labels

app_labels.groupby(["app_id"], as_index=False).count().sort_values(by="label_id", ascending=False)
label_categories[label_categories.label_id.isin(app_labels[app_labels.app_id == 6792270137491452041].label_id)]
print("duplicate labels?: " + str(label_categories.shape[0]-len(label_categories.label_id.unique())))

print("duplicate events?: " + str(events.shape[0]-len(events.event_id.unique())))
# denormalized events

events_deno = events.merge(app_events, on="event_id")

events_deno = events_deno.drop(["event_id", "longitude", "latitude", "is_installed"], axis=1)

events_deno['used_at'] = events_deno.timestamp.dt.hour
phone_events = phones.merge(events_deno, on='device_id')

len(phone_events)
print(len(phones.device_id.unique()))

print(len(phones.device_id) - len(phones.device_id.value_counts()))
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(gatrain.group)

n_classes = len(le.classes_)

pred = np.ones((gatrain.shape[0],n_classes))/n_classes

pred
phones.head()

phones.groupby(["phone_brand", "device_model"], as_index=False).count()
phone_events = phone_events.merge(app_labels, on="app_id")
