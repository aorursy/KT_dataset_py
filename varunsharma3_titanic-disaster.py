# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
titanic_df = pd.read_csv('../input/train.csv')
titanic_df.head()
titanic_df.info()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.factorplot('Pclass',data=titanic_df,kind='count',hue='Sex')
def male_female_child(passenger):
    age,sex = passenger
    if age<16:
        return 'child'
    else:
        return sex
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
sns.factorplot('Person',data=titanic_df,hue='Survived',kind='count')
titanic_df['Person'].value_counts()
fig = sns.FacetGrid(titanic_df,hue='Person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
fig.add_legend()
fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
fig.add_legend()
titanic_df.head()
decks = titanic_df['Cabin'].dropna()
levels = []
for deck in decks:
    levels.append(deck[0])
levels

cabin_df = pd.DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d',kind='count')
cabin_df = cabin_df[cabin_df.Cabin!='T']
sns.factorplot('Cabin',data=cabin_df,palette='summer_d',kind='count')
#Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
sns.factorplot('Embarked',data=titanic_df,kind='count',hue='Pclass',palette='summer_d')
sns.factorplot('Survived',data=titanic_df[titanic_df.Embarked=='Q'],hue='Sex',kind='count',palette='winter_d')
#Who was alone and who was with family
# alone means some one who doesnt have a parent or children i.e Parch=0 and Sibsp=0(1 meanssomeone was with them)
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df.head()
sns.factorplot('Pclass','Survived',data=titanic_df,hue='Person')
sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass',palette='winter')
generations = [10,20,30,40,50,60,70]
sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass',palette='winter',x_bins=generations)
sns.lmplot('Age','Survived',data=titanic_df,hue='Sex',palette='winter')
sns.lmplot('Age','Survived',data=titanic_df,hue='Embarked',palette='coolwarm')
def withorwithout(person):
    SiblingOrSpoouse,Children = person
    if (SiblingOrSpoouse > 0 or Children >0):
        return 'With Family'
    else:
        return 'Without Family'
titanic_df['WithORWithoutFamily'] = titanic_df[['SibSp','Parch']].apply(withorwithout,axis=1)
sns.factorplot('WithORWithoutFamily','Survived',data=titanic_df,hue='Sex')
#Final analysis -

#Drop all rows with missing values
titanic_df.dropna(inplace=True)

# Convert Categorical features into numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder
Sex_le = LabelEncoder()
titanic_df['Sex']=Sex_le.fit_transform(titanic_df['Sex'])

Person_le = LabelEncoder()
titanic_df['Person'] = Person_le.fit_transform(titanic_df['Person'])

WithORWithoutFamily_le = LabelEncoder()
titanic_df['WithORWithoutFamily'] = Person_le.fit_transform(titanic_df['WithORWithoutFamily'])

#Get Dummies for Embarked
temp = pd.get_dummies(titanic_df['Embarked'],prefix='Station')
titanic_df = pd.concat([titanic_df,temp],axis=1)

#Drop columns which are of no value to the model or logically otherwise.
titanic_df = titanic_df[['Pclass','Survived','Sex','Age','WithORWithoutFamily','Station_C','Station_Q','Station_S','Person']]
titanic_df.head()
X = titanic_df[['Pclass','Sex','Age','WithORWithoutFamily','Station_C','Station_Q','Station_S','Person']]
Y = titanic_df[['Survived']]
names = X.columns

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X,Y)

#Show features by importance
print('Features sorted by their scores')
temp = sorted(zip(map(lambda x:np.round(x,4),rf.feature_importances_),names),reverse=True)
print(temp)
feature = pd.DataFrame(temp)
feature.columns = [['Value','FeatureName']]
sns.boxplot(x='Value',y='FeatureName',data=feature)
plt.figure(figsize=(8,8))
corr =  titanic_df.corr()

#Create a mask for hidding duplicate blocks
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True

#Draw heatmap
sns.heatmap(corr,cmap='coolwarm',annot=True,mask=mask)
#Age, Sex and Person have the highest impact on survivability. 
