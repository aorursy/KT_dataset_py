# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

i = 0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if i < 10:

            print(os.path.join(dirname, filename))

            i = i + 1



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf

from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm

from skimage import io
train_data = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test_data = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
train_data.shape, test_data.shape
train_data.head(5)
test_data.head(3)
print("Number of patients in train dataset : ",train_data['patient_id'].nunique())

print("Number of patients in test dataset : ",test_data['patient_id'].nunique())

common_data = pd.merge(train_data,test_data,on=['patient_id','patient_id'])

print("Number of common patients in train and test dataset : ",common_data['patient_id'].nunique())
train_data['target'].value_counts()
train_data['age_approx'].nunique()
train_data['anatom_site_general_challenge'].value_counts()
f, ax = plt.subplots(1,2, figsize=(18,6))



sns.countplot(x='sex',data=train_data,hue='target',ax=ax[0])

sns.countplot(x='anatom_site_general_challenge',data=train_data,hue='target',ax=ax[1])
f, ax1 = plt.subplots(1,1,figsize=(18,6)) 



sns.countplot(x='age_approx',data=train_data,hue='target',ax=ax1)
df = train_data.copy()



df1 = df.loc[df['target'] == 1][['age_approx']].groupby('age_approx').agg({'age_approx':'count'})

df2 = df.loc[df['target'] == 0][['age_approx']].groupby('age_approx').agg({'age_approx':'count'})

df1.columns = ['count']

df2.columns = ['count']

df1.loc[0] = [0]

df1.loc[10] = [0]

df1 = df1.sort_values('age_approx')

df1['total'] = df1['count'] + df2['count']

df1['perc_malignant'] = df1['count']/df1['total']

df1 = df1.reset_index()

f, ax2 = plt.subplots(1,1,figsize=(18,6))



sns.barplot(x='age_approx',y='perc_malignant',data=df1,ax=ax2)

ax2.set_title('Percent malignant in age approx');
image_list = train_data[train_data['target'] == 0].sample(8)['image_name']

image_all=[]

for image_id in image_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    image_all.append(img)
f, ax = plt.subplots(2,4,figsize=(18,8))



c = 0

for i in range(2):

    for j in range(4):

        ax[i][j].imshow(image_all[c])

        ax[i][j].axis('off')

        c = c + 1
melanoma_list = train_data[train_data['target'] == 0].sample(8)['image_name']

melanoma_all=[]

for image_id in melanoma_list:

    image_file = f'/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+image_id+'.jpg' 

    img = np.array(Image.open(image_file))

    melanoma_all.append(img)
f, ax1 = plt.subplots(2,4,figsize=(18,8))



c = 0

for i in range(2):

    for j in range(4):

        ax1[i][j].imshow(melanoma_all[c])

        ax1[i][j].axis('off')

        c = c + 1
train_data.isna().sum(), test_data.isna().sum()
na_cols = train_data.columns[train_data.isna().any()].tolist()
for col in na_cols:

    mode = train_data[col].mode().values[0]

    train_data[col] = train_data[col].fillna(mode)

    test_data[col] = test_data[col].fillna(mode)
train_data.isna().sum(), test_data.isna().sum()
datasets = [train_data, test_data]



for df in datasets:

    df['sex_label'] = np.where(df['sex']=='female',1,0)
train_data['anatom_site_general_challenge'].value_counts()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train_data['anatom_label'] = le.fit_transform(train_data['anatom_site_general_challenge'].astype('str'))

test_data['anatom_label'] = le.transform(test_data['anatom_site_general_challenge'].astype('str'))

train_data.isna().sum(), test_data.isna().sum()
train_data.describe()
train_images = train_data['image_name'].values

train_sizes = np.zeros(train_images.shape[0])

for i, img_path in enumerate(tqdm(train_images)):

    train_sizes[i] = os.path.getsize(os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/', f'{img_path}.jpg'))

    

train_data['image_size'] = train_sizes





test_images = test_data['image_name'].values

test_sizes = np.zeros(test_images.shape[0])

for i, img_path in enumerate(tqdm(test_images)):

    test_sizes[i] = os.path.getsize(os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/', f'{img_path}.jpg'))

    

test_data['image_size'] = test_sizes
from sklearn.preprocessing import MinMaxScaler



minmax = MinMaxScaler()



train_data['image_size_scaled'] = minmax.fit_transform(train_data['image_size'].values.reshape(-1,1))

test_data['image_size_scaled'] = minmax.transform(test_data['image_size'].values.reshape(-1,1))
from sklearn.preprocessing import KBinsDiscretizer

categorize = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')

train_data['image_size_enc'] = categorize.fit_transform(train_data.image_size_scaled.values.reshape(-1, 1)).astype(int).squeeze()

test_data['image_size_enc'] = categorize.transform(test_data.image_size_scaled.values.reshape(-1, 1)).astype(int).squeeze()
plt.figure(figsize = (12,6))

sns.countplot(x = 'image_size_enc', hue = 'target', data = train_data)
# Function for finding average colour of an image. 

# Didn't need to use it after adding data from mean color isic 2020



def average_color(image_path):

    from skimage import io

    img = io.imread(image_path)[:, :, :-1]

    avg = img.mean(axis=0).mean(axis=0).mean()

    return avg
# It was taking too long to calculate the colour values myself. Uncomment for using the code



#avg_colors_train = np.zeros(train_data.shape[0])



#for i, img_path in enumerate(tqdm(train_images)):

#    image_path = os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/', f'{img_path}.jpg')

#    avg = average_color(image_path)

#    avg_colors_train[i] = avg

    

#train_image_color['avg_color'] = avg_colors_train



#train_image_color.to_csv('/kaggle/working/created_data/train_image_color.csv',index=False)'''
# For test data

#avg_colors_test = np.zeros(test_data.shape[0])



#for i, img_path in enumerate(tqdm(test_images)):

#    image_path = os.path.join('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/', f'{img_path}.jpg')

#    avg = average_color(image_path)

#    avg_colors_test[i] = avg

    

#test_image_color['avg_color'] = avg_colors_test



#test_image_color.to_csv('/kaggle/working/created_data/test_image_color.csv',index=False)
train_colors = pd.read_csv('/kaggle/input/mean-color-isic2020/train_color.csv')

test_colors = pd.read_csv('/kaggle/input/mean-color-isic2020/test_color.csv')



train_data['avg_color'] = train_colors['color_mean']

test_data['avg_color'] = test_colors['color_mean']
train_data.groupby('patient_id').agg({'age_approx':'min'}).reset_index().head()
df_min = train_data.groupby('patient_id').agg({'age_approx':'min'}).reset_index()

df_min.columns = ['patient_id','age_min_id']

train_data = pd.merge(train_data, df_min, on = 'patient_id', how= 'left')



df_max = train_data.groupby('patient_id').agg({'age_approx':'max'}).reset_index()

df_max.columns = ['patient_id','age_max_id']

train_data = pd.merge(train_data, df_max, on = 'patient_id', how= 'left')
df_min = test_data.groupby('patient_id').agg({'age_approx':'min'}).reset_index()

df_min.columns = ['patient_id','age_min_id']

test_data = pd.merge(test_data, df_min, on = 'patient_id', how= 'left')



df_max = test_data.groupby('patient_id').agg({'age_approx':'max'}).reset_index()

df_max.columns = ['patient_id','age_max_id']

test_data = pd.merge(test_data, df_max, on = 'patient_id', how= 'left')
train_data.head()
train_X = train_data[test_data.describe().columns]

train_y = train_data[['target']]
train_X.head(3)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)



rf.fit(train_X,train_y)

train_proba = rf.predict_proba(train_X)[:,1]

test_proba = rf.predict_proba(test_data[train_X.columns])[:,1]

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(train_y, train_proba)
def plot_roc_curve(fpr, tpr):

    plt.figure(figsize=(12,10))

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
plot_roc_curve(fpr,tpr)
roc_auc_score(train_y, train_proba)
rf = RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state = 42)

rf.fit(train_X, train_y)
predictions = []

for tree in rf.estimators_:

    predictions.append(tree.predict_proba(train_X)[None, :])
predictions = np.vstack(predictions)

cum_mean = np.cumsum(predictions, axis=0)/np.arange(1, predictions.shape[0] + 1)[:, None, None]



scores = []

for pred in cum_mean:

    scores.append(roc_auc_score(train_y, np.argmax(pred, axis=1)))
plt.figure(figsize=(10, 6))

plt.plot(scores, linewidth=3)

plt.xlabel('num_trees')

plt.ylabel('roc_auc');
features_list = [5,7,9]

depth_list = [3,5,8]
# Hyperparameter tuning

import time



n_trees = 100

models = {}

roc_train_dict = {}



for max_features in features_list:

    for max_depth in depth_list:

        start = time.time()

        model = RandomForestClassifier(n_estimators = n_trees, max_depth = max_depth, max_features=max_features, n_jobs = -1, random_state=42)

        model.fit(train_X,train_y)

        train_pred = model.predict_proba(train_X)[:,1]

        train_roc = roc_auc_score(train_y,train_pred)

        roc_train_dict[max_features,max_depth] = round(train_roc,3)

        models[max_features,max_depth] = model

        end = time.time()

        time_taken = round(end-start,3)

        print("Time taken for ",max_depth," depth and ",max_features," features is : ",time_taken," seconds.")
roc_df = pd.DataFrame(roc_train_dict, index=['train_roc']).transpose()

roc_df = roc_df.reset_index()

roc_df.columns = ['max_features','max_depth','train_roc']

roc_df.head(20)
max_features = roc_df[roc_df['train_roc'] == roc_df['train_roc'].max()]['max_features'].values[0]

max_depth = roc_df[roc_df['train_roc'] == roc_df['train_roc'].max()]['max_depth'].values[0]

model = models[max_features,max_depth]
submissions = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
submissions.head()
test_proba = model.predict_proba(test_data[train_X.columns])[:,1]
submissions.shape, test_proba.shape
submissions['target'] = test_proba
submissions.to_csv('/kaggle/working/submissions_melanoma.csv',index=False)