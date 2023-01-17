import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss

from sklearn.cross_validation import KFold

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
age_train = pd.read_csv('../input/gender_age_train.csv')
age_train.head()
fig,ax = plt.subplots(figsize=(8,6))



age_train.group.value_counts().plot(kind='barh')

plt.title('Age Group Distribution')

plt.xlabel('Count')
fig,ax = plt.subplots(figsize=(8,6))



age_train.gender.value_counts().plot(kind='barh')

plt.title('Gender Distribution')

plt.xlabel('Count')

plt.ylabel('Gender')
fig,ax = plt.subplots(figsize=(8,6))

plt.title('Age Distribution')

sns.distplot(age_train.age,ax=ax)
le = LabelEncoder().fit(age_train.group.values)

y = le.transform(age_train.group.values)

n_classes = len(le.classes_)

print('Class Length : ',len(le.classes_))
pred = np.ones((age_train.shape[0],n_classes)) / n_classes

print('log loss benchmark : {}'.format(log_loss(y,pred)))
n_folds = 5

kf = KFold(age_train.shape[0],n_folds=n_folds,shuffle=True,random_state=42)

preds = np.zeros((age_train.shape[0],n_classes))

for i,(train_idx,test_idx) in enumerate(kf):

    X_train,X_valid = age_train.iloc[train_idx,:],age_train.iloc[test_idx,:]

    y_train,y_valid = y[train_idx],y[test_idx]

    prob = X_train.groupby('group').size() / X_train.shape[0]

    preds[test_idx,:] = prob

log_loss(y,preds)
phone = pd.read_csv('../input/phone_brand_device_model.csv',encoding='utf-8')
phone.head()
## duplicate

phone.shape[0] - phone.device_id.nunique()
phone_size = phone.groupby('device_id').size()

phone_idx = phone_size[phone_size > 1]

len(phone_idx) ## duplicate
phone.loc[phone.device_id.isin(phone_idx.index),:].head()
count = phone.groupby(['device_model'])['phone_brand'].apply(pd.Series.nunique)

count.value_counts()
lebrand = LabelEncoder().fit(phone.phone_brand.values)

phone['brand'] = lebrand.transform(phone.phone_brand.values)

m = phone.phone_brand.str.cat(phone.device_model)

lemodel = LabelEncoder().fit(m)

phone['model'] = lemodel.transform(m)
phone.head()
phone['model'].plot(kind='hist',bins=50)

plt.title('Model Distribution')

plt.xlabel('Model')
phone['brand'].plot(kind='hist',bins=50)

plt.title('Brand Distribution')

plt.xlabel('Brand')
events = pd.read_csv('../input/app_events.csv')
events.head()
events['is_active'].value_counts().plot(kind='bar')
events.shape[0] - events['is_installed'].value_counts()
events['event_id'].value_counts().plot(kind='hist',bins=50)