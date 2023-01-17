# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#1. Reading Data



df_test=pd.read_csv("../input/sanbercode/test.csv")

df_train=pd.read_csv("../input/sanbercode/train.csv")

#Looking into training data

df_train.info()

print(df_train.columns.tolist())

print("Jumlah kolom: {}".format(len(df_train.columns.tolist())))
#Looking into test data

df_test.info()
#2. Data visualization

#Splitting into cat and nums



df_num=df_train[['Umur','Jmlh Tahun Pendidikan','Jam per Minggu']]

df_cat=df_train[['Kelas Pekerja','Pendidikan','Status Perkawinan','Pekerjaan','Jenis Kelamin','Gaji']]
#Plotting data

#Numericals as Histograms

import seaborn as sns

import matplotlib.pyplot as plt

for i in df_num.columns:

    plt.hist(df_num[i])

    plt.title(i)

    plt.show()
#Correlation Values

print(df_num.corr)

sns.heatmap(df_num.corr())
#Gaji Across Numeric Data

pd.pivot_table(df_train,index='Gaji',values=df_num)
#Plotting Cat data

for i in df_cat.columns:

    plt.figure(figsize=(12,8))

    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)

    plt.show()
#Salary rates

pd.pivot_table(df_train,index='Gaji',columns='Kelas Pekerja',values='Umur',aggfunc='count')
pd.pivot_table(df_train,index='Gaji',columns='Jenis Kelamin',values='Umur',aggfunc='count')
pd.pivot_table(df_train,index='Gaji',columns='Status Perkawinan',values='Umur',aggfunc='count')
pd.pivot_table(df_train,index='Gaji',columns='Pendidikan',values='Umur',aggfunc='count')
pd.pivot_table(df_train,index='Gaji',columns='Pekerjaan',values='Umur',aggfunc='count')
#3. Data Normalization

df_train.Umur=np.log(df_train.Umur+1)



df_test.Umur=np.log(df_test.Umur+1)
df_train.Umur.hist()
#4. Model Preprocessing

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scale=StandardScaler()

mms=MinMaxScaler()



from sklearn.model_selection import train_test_split

#Dropping values which do not yield better results, swapping gender and salary with binary values

df_train_2=df_train.drop(['Berat Akhir','Kerugian Capital','Keuntungan Kapital','id'],axis=1).replace({'Perempuan':0,'Laki2':1}).replace({'<=7jt':0,'>7jt':1})

df_train_2=pd.get_dummies(df_train_2,columns=['Status Perkawinan', 'Pekerjaan','Pendidikan','Kelas Pekerja'])



X=df_train_2.drop('Gaji',axis=1)

y=df_train_2.Gaji



#Tuning X

X=pd.DataFrame(mms.fit_transform(X),columns=X.columns)

X.Umur = pd.DataFrame(scale.fit_transform(pd.DataFrame(X.Umur)))



#Tuning test

df_test_2=df_test.drop(['Berat Akhir','Kerugian Capital','Keuntungan Kapital','id'],axis=1).replace({'Perempuan':0,'Laki2':1}).replace({'<=7jt':0,'>7jt':1})

df_test_2=pd.get_dummies(df_test_2,columns=['Status Perkawinan', 'Pekerjaan','Pendidikan','Kelas Pekerja'])

df_test_2=pd.DataFrame(mms.transform(df_test_2), columns=df_test_2.columns)

df_test_2.Umur = pd.DataFrame(scale.transform(pd.DataFrame(df_test_2.Umur)))



X_train,X_test,y_train,y_test=train_test_split (X, y, test_size=0.25,stratify=y, random_state=17)
X
#5. Model Tuning

#5a. Grid Search for best parameters

#Already chose AdaBoost after several attempts on different models (LR,RF,XGB,SVC,etc. Mostly yield sub-89%, AdaBoost yields 90%+)

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

def clf_performance(classifier, model_name):

    print(model_name)

    print('Best Score: ' + str(classifier.best_score_))

    print('Best Parameters: ' + str(classifier.best_params_))



ada = AdaBoostClassifier()

param_grid = {

    'n_estimators': [100,250,200,225],

    'learning_rate':[.01,0.1,0.2,0.3,0.5, 0.7, 0.9],

    'random_state' : [0,1]

}

'''clf_ada = GridSearchCV(ada, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

best_clf_ada=clf_ada.fit(X_train, y_train)



clf_performance(best_clf_ada,'Ada Boost')

Ada Boost

Best Score: 0.836228931283571

Best Parameters: {'learning_rate': 0.3, 'n_estimators': 250, 'random_state': 0}'''

#5b. Plugging in best parameters

#Copied best parameters from previous best,

ada=AdaBoostClassifier(learning_rate=0.2,n_estimators=200,random_state=0)

ada.fit(X,y)

pre_final=ada.predict(df_test_2)

df_test_2['Gaji']=pre_final
#6. Submission

hasil_gaji=df_test_2['Gaji']

ada_submission={'id':df_test.id,'Gaji':hasil_gaji}

submission_ada=pd.DataFrame(data=ada_submission)

submission_ada.to_csv('ada_submission.csv',index=False)