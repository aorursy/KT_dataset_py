# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#Load the csv files 
os.chdir("../input")
df1=pd.read_csv('heroes_information.csv')
df2=pd.read_csv('super_hero_powers.csv')
df1.info()
df1.head()
df2.info()
df2.head()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('dark')
plt.subplots(figsize=(20,10))
ax=sns.countplot(x='Publisher',data=df1,palette='husl')
ax.set_xticklabels(ax.get_xticklabels(),rotation=40)
df1=df1.drop("Unnamed: 0",axis=1) #dropping unnecessary columns
#Replacing hyphen with 'unknown' for missing values
labels=['Gender','Eye color','Race','Hair color','Skin color','Alignment','Publisher']
df1[labels]=df1[labels].replace('-','Unknown')

df1[labels]=df1[labels].apply(lambda x:x.astype('category'),axis=0)
unique_cat=df1[labels].apply(pd.Series.nunique,axis=0)
unique_cat.plot(kind='bar',figsize=(10,10))
df1.describe()
sns.pairplot(data=df1,vars=['Height','Weight'],kind='scatter',hue='Gender',palette='bright',diag_kws={'alpha':.5})
#calculate body mass index using height and weight
df1['Height']=df1['Height']*0.393701     #converting cm to inches
df1['bmi']=(df1['Weight']*703)/(df1['Height']**2)

df1.loc[(df1.bmi<0),'body_type']='Unknown'
df1.loc[(df1.bmi>0) & (df1.bmi<16),'body_type']='Severe thinness'
df1.loc[(df1.bmi>16) & (df1.bmi<=17),'body_type']='Moderate thinness'
df1.loc[(df1.bmi>17) & (df1.bmi<=18.5),'body_type']='Mild thinness'
df1.loc[(df1.bmi>18.5) & (df1.bmi<=25),'body_type']='Normal'
df1.loc[(df1.bmi>25) & (df1.bmi<=30),'body_type']='Average'
df1.loc[(df1.bmi>30) & (df1.bmi<=40),'body_type']='Overweight'
df1.loc[(df1.bmi>=40),'body_type']='Obese'

sns.set_style("whitegrid")
plt.subplots(figsize=(10,5))
sns.heatmap(pd.crosstab(df1['Alignment'],df1['Gender']))
plt.subplots(figsize=(15,5))
sns.countplot(x='body_type',data=df1,palette='Paired')
#Examining race and body type
sns.set_style("dark")
plt.subplots(figsize=(10,10))
sns.heatmap(pd.crosstab(df1['Race'],df1['body_type']),cmap="PiYG")
df2.head()
df1['name'].duplicated().sum()
df1=df1.drop_duplicates(subset='name')
df=df1.set_index('name').join(df2.set_index('hero_names'))
df=df.drop(df1.drop(['name','Alignment'],axis=1).columns,axis=1)
df.head()
df=df.dropna(thresh=2) #dropping all rows with more than 2 missing values
dfnew=df[~(df['Alignment']=='Unknown')]  #Create a dataset where the alignment is known (this data is used for running and testing the model)
new_data=df[df['Alignment']=='Unknown'] #Create a dataset where the alignment is unknown (Let's set aside this data)
from sklearn.model_selection import train_test_split #importing train test split to split the data for validation

X=dfnew.drop('Alignment',axis=1)
y=dfnew['Alignment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier

neighbors=np.arange(1,20)           #setting the possible neighbor values
train_acc=np.empty(len(neighbors))   #Initialize empty array for storing training accuracy
test_acc=np.empty(len(neighbors))    #Initialize empty array for storing training accuracy

    
#Run a for loop with KNN model for each neighbor and store the training and testing accuracy for each iteration
for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    train_acc[i]=knn.score(X_train,y_train)
    test_acc[i]=knn.score(X_test,y_test)
    
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_acc, label = 'Testing Accuracy')
plt.plot(neighbors, train_acc, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

newmodel=KNeighborsClassifier(n_neighbors=13)
newmodel.fit(X_train,y_train)

newmodel.score(X_test,y_test)
new_data['Alignment_predict']=newmodel.predict(new_data.drop('Alignment',axis=1))
new_data['Alignment_predict']











  


