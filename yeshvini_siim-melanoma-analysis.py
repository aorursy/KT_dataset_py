# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#      for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## All files

path = '../input/siim-isic-melanoma-classification'

print(os.listdir(path))
train_df=pd.read_csv(path+'/train.csv')

test_df=pd.read_csv(path+'/test.csv')

print("Train -",train_df.shape)

print("Test - ",test_df.shape)

print('Train Features',train_df.columns)

print('Test Features',test_df.columns)
train_df.head()
missing_value_df_train = pd.DataFrame(index = train_df.keys(), data =train_df.isnull().sum(), 

                                      columns = ['Missing_Value_Count'])

missing_value_df_train['Missing_Value_Percentage'] = ((train_df.isnull().mean())*100)

missing_value_df_train.sort_values('Missing_Value_Count',ascending= False)


# for cols in columns:

#     train_df[cols].fillna('na',inplace=True)



# Replace age na values with median

age = train_df[train_df["age_approx"]!=np.nan]["age_approx"]

train_df["age_approx"].replace(np.nan, age.median(), inplace=True)



# Replace sex & anatom_site na values with mode

columns=['anatom_site_general_challenge','sex']

for cols in columns:

    na = train_df[train_df[cols]!=np.nan][cols]

    train_df[cols].replace(np.nan, na.mode().values[0], inplace=True)



train_df.isnull().sum()
train_df.groupby('benign_malignant')['sex'].value_counts().plot(kind='bar')
train_df.groupby(['sex','target'])['benign_malignant'].count().to_frame().reset_index()
sns.countplot(data=train_df,x='sex')
train_df.diagnosis.value_counts()
print('Count of patient ids ( including duplicates) - ',train_df.patient_id.count())

print('Count of unique patient ids in train set - ',train_df.patient_id.nunique())

train_patient_unique=train_df[train_df.target==1]

print('Number of Patients diagnosed with melanoma - ',len(train_patient_unique))
plt.figure(figsize=(8,8))

sns.countplot(data=train_patient_unique,x='anatom_site_general_challenge',hue='sex')
uniq_ids=train_patient_unique[train_patient_unique.duplicated(['patient_id'])]

len(np.array(uniq_ids))

train_patient_unique[train_patient_unique.anatom_site_general_challenge == 'oral/genital']
#np.array(uniq_ids.patient_id)
uniq_ids[uniq_ids.patient_id == 'IP_9086201']

#uniq_ids[uniq_ids.patient_id == 'IP_5399626']
#train_df.age_approx.value_counts().plot(kind='bar')

plt.figure(figsize=(8,8))

sns.countplot(data=train_patient_unique,x='age_approx',hue='sex')
benign = train_df.image_name[train_df['benign_malignant']=='benign']

malignant = train_df.image_name[train_df['benign_malignant']=='malignant']
def viz(images):

    # Plot first 10 images

    image_list=[i+'.jpg' for i in images]

    image_list= image_list[:10]

    img_dir = path+'/jpeg/train'

    plt.figure(figsize=(8,8))

    # Iterate and plot random images

    for i in range(9):

        plt.subplot(3,3,i+1)

        img = plt.imread(os.path.join(img_dir, image_list[i]))

        plt.imshow(img)

        plt.axis('off')

    plt.tight_layout()
viz(benign)
viz(malignant)