# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



input_dir="../input/titanic"

print(os.listdir(input_dir))



# Any results you write to the current directory are saved as output.
data=pd.read_csv(input_dir+'/train.csv',header=0)

#data=data.dropna()

data.head()
data.isna().sum()
def fix_nan(data_df):

    #Age

    for name_string in data_df['Name']:

        data_df['Title']=data_df['Name'].str.extract('([A-Za-z]+)\.',expand=True)

    

    #replacing the rare title with more common one.

    mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Rev':'Mr','Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

              'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

    data_df.replace({'Title': mapping}, inplace=True)



    titles=['Mr','Miss','Mrs','Master','Dr']

    for title in titles:

        age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]

        #print(age_to_impute)

        data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

    

    #fix any remaining nan Age with median

    data_df.loc[(data_df['Age'].isnull()) , 'Age']= data_df['Age'].median()

    

    #Fare, if inneed

    Pclasss=data_df['Pclass'].unique()

    Pclasss=Pclasss.tolist();

    for pc in Pclasss:

        Fare_to_impute = data_df.groupby('Pclass')['Fare'].median()[pc]

        #print(age_to_impute)

        data_df.loc[(data_df['Fare'].isnull()) & (data_df['Pclass'] == pc), 'Fare'] = Fare_to_impute

    

    #Embark, with "S"

    data.loc[(data['Embarked'].isnull()) , 'Embarked']= "S"

    

    

    return data_df
data=fix_nan(data)
sexMap={'female':0,'male':1}

data=data.replace({'Sex':sexMap})
data['Embarked'].unique()
embarkMap={'S':1,'C':2,'Q':3}

data=data.replace({'Embarked':embarkMap})
# from: A Data Science Framework: To Achieve 99% Accuracy

def correlationMapPlot(data):

    

    import seaborn as sns



    plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    sns.heatmap(

        data.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlationMapPlot(data)
plt.hist(x=data.loc[(data['Survived']==1) , 'Age'], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
#collection of some common feature engineering practices founded

def featureEngineering(data):    

    data['isInfant'] = data ['Age']<5;

    

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    

    data['isAlone'] = 0

    data.loc[data['FamilySize'] == 1, 'isAlone'] = 1

    #IndividualFare can be dependent to FamilySize

    data['IndividualFare']=(data['Fare'])/(data ['SibSp'] +data['Parch']+1);



    

    return data
data=featureEngineering(data)
correlationMapPlot(data)
X = np.asarray(data[['Sex','Pclass','Fare','SibSp','Parch','Embarked','isInfant','isAlone','FamilySize','IndividualFare']])

y = np.asarray(data[['Survived']])



from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

import tensorflow as tf

from tensorflow import keras
model = keras.Sequential([

  keras.layers.Dense(16, activation='relu', input_shape=(10,)),

  keras.layers.Dense(32, activation='relu'),

  keras.layers.Dense(16, activation='relu'),

  keras.layers.Dense(1, activation='sigmoid'),

])



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)
yhat=model.predict(X_test)
yhat=model.predict(X_test)

for i in range(0,len(yhat)):

    if yhat[i]>=0.6:

        yhat[i]=1;

    else:

        yhat[i]=0;   

        


from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y_test, yhat)
data_fin.isna().sum()
data_fin=pd.read_csv(input_dir+'/test.csv',header=0)

data_fin_1=data_fin.replace({'Sex':sexMap})

data_fin_1=data_fin_1.replace({'Embarked':embarkMap})

data_fin_1=fix_nan(data_fin_1);

data_fin_1=featureEngineering(data_fin_1);

X_fin = np.asarray(data_fin_1[['Sex','Pclass','Fare','SibSp','Parch','Embarked','isInfant','isAlone','FamilySize','IndividualFare']])

X_fin = preprocessing.StandardScaler().fit(X_fin).transform(X_fin)

data_fin_1
yhat_fin = model.predict(X_fin)

for i in range(0,len(yhat_fin)):

    if yhat_fin[i]>=0.6:

        yhat_fin[i]=1;

    else:

        yhat_fin[i]=0;  
yhat_fin=yhat_fin.astype(int)
yhat_fin = yhat_fin.flatten()
yhat_fin.shape
submission_dat=pd.DataFrame()

submission_dat.loc[:,'PassengerId']=data_fin['PassengerId']

submission_dat.loc[:,'Survived']=pd.Series(yhat_fin, index=submission_dat.index)
submission_dat['Survived'].unique()
submission_dat.to_csv('sampleSubmission.csv',index=False)