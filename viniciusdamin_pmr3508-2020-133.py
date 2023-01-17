import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df=pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=["Id"],na_values='?')



#usado para termos uma noção de como é a tabela

df.head()



df.info()
df.describe()
#tabela com a variável e quantos dados faltantes tem

total=df.isnull().sum().sort_values(ascending= False)



#tabela com porcentagem de dados faltantes de cada variável

percent=(df.isnull().sum()*100/df.notnull().count()).sort_values(ascending= False)



#tabela que junta as duas anteriores

missing_data=pd.concat([total, percent],axis=1, keys=['total',"%"])

missing_data.head(4)

a=df.occupation.describe()

b=df.workclass.describe()

c=df['native.country'].describe()

missing_data_analysis=pd.concat([a,b,c],axis=1)

missing_data_analysis.head()

for column in ['workclass', 'native.country', 'occupation']:

    missing_data=df[column].mode()[0]

    df[column]=df[column].fillna(missing_data)

#tabela com a variável e quantos dados faltantes tem

total=df.isnull().sum().sort_values(ascending= False)



#tabela com porcentagem de dados faltantes de cada variável

percent=(df.isnull().sum()*100/df.notnull().count()).sort_values(ascending= False)



#tabela que junta as duas anteriores

missing_data=pd.concat([total, percent],axis=1, keys=['total',"%"])

missing_data.head(4)
a=df.occupation.describe()

b=df.workclass.describe()

c=df['native.country'].describe()

missing_data_analysis=pd.concat([a,b,c],axis=1)

missing_data_analysis.head()
df.drop_duplicates(keep="first", inplace=True)
df.describe()
sns.set_context('poster')

sns.pairplot(df,vars=['age', 'fnlwgt', 'education.num', 'hours.per.week'], hue='income')
sns.set_context('poster')

#sns.pairplot(df,vars=['age', 'fnlwgt', 'education.num', 'hours.per.week'], kind='kde', hue='income')
from sklearn.preprocessing import LabelEncoder



df['income']=LabelEncoder().fit_transform(df['income'])



df['income'].head()
df.describe()
plt.figure(figsize=(10,10))

sns.set_context('notebook')

sns.heatmap(df.corr(), square = True, annot=True, vmin=-1, vmax=1)

plt.show()
#histograma

plt.figure(figsize=(11,5))

df['age'].hist(bins=15)

plt.xlabel('age')

plt.ylabel('number of people')

plt.title("Age histogram")



#curva de distribuição

sns.set()

sns.set_context('notebook')

#sns.displot(df["age"],kind='kde', bw_adjust=0.7, height=4, aspect=2.5)
sns.set_context('notebook')

#sns.displot(df, x='age', hue='income' ,kind='kde',height=4,fill=True,aspect=2.5)
sns.set_context('notebook')

#sns.displot(df, x='fnlwgt', hue='income' ,kind='kde',height=4,fill=True,aspect=2.5)
sns.set_context('notebook')

#sns.displot(df, x='education.num', hue='income' ,kind='kde',height=4,fill=True,aspect=2.5)
sns.set_context('notebook')

#sns.displot(df, x='capital.gain', hue='income' ,kind='hist',height=4,fill=True,aspect=2.5)
df_proof=df['capital.gain'].sort_values(ascending=False)

df_proof.head(163)
df_smaller=df.loc[df['capital.gain']<99999]



sns.set_context('notebook')

#sns.displot(df_smaller, x='capital.gain', hue='income' ,kind='hist',height=4,fill=True,aspect=2.5)
df_smaller['capital.gain'].describe()
df_in_between=df_smaller.loc[df['capital.gain']>0]

#sns.histplot(df_try,x='capital.gain',hue='income')

#sns.displot(df_in_between, x='capital.gain', hue='income' ,kind='hist',height=4,fill=True,aspect=2.5)


print('Temos',len(df_smaller.loc[df['capital.gain']==0]), 'observações onde "capital.gain=0"')

print('Temos',len(df_smaller.loc[df['capital.gain']==0].loc[df['income']==1]), 'observações onde "capital.gain=0" e que "income">50K ')

print('Temos',len(df_smaller.loc[df['capital.gain']>0]), 'observações onde "capital.gain>0"')

print('Temos',len(df_smaller.loc[df['capital.gain']>0].loc[df['income']==1]), 'observações onde "capital.gain>0" e que "income">50K ')

#sns.displot(df,x='capital.loss',hue='income',kind='hist',height=4,fill=True,aspect=2.5)
#sns.displot(df,x='hours.per.week',hue='income',kind='kde', height=4,fill=True,aspect=2.5)
sns.catplot(data=df, x='income', y='workclass', kind='bar')
sns.catplot(data=df,x='income',y='education', kind='bar')
sns.catplot(data=df,x='income',y='marital.status',kind='bar')
sns.catplot(data=df,x='income',y='occupation',kind='bar')
sns.catplot(data=df,x='income',y='relationship',kind='bar')
sns.catplot(data=df,x='income',y='race',kind='bar')
sns.catplot(data=df, x='income', y='sex', kind='bar')
sns.catplot(data=df, x='income', y='native.country', kind='bar')
#sns.displot(df, y='native.country', height=8)
df_new=df.drop(['education','fnlwgt','native.country'], axis='columns')

df_new.head()

y_train=df_new['income']

X_train=df_new.drop(['income'],axis='columns')
numerical_columns=list(X_train.select_dtypes(include=[np.number]))

numerical_columns.remove('capital.gain')

numerical_columns.remove('capital.loss')



categorical_columns=list(X_train.select_dtypes(exclude=[np.number]))



capital_columns=['capital.gain','capital.loss']



print("numerical_columns=",numerical_columns)

print("categorical_columns=",categorical_columns)

print("capital_columns=",capital_columns)
from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()
from sklearn.preprocessing import RobustScaler

robust_scaler=RobustScaler()
from sklearn.preprocessing import OneHotEncoder

hot_encode=OneHotEncoder(sparse=False)
from sklearn.compose import ColumnTransformer

column_transf=ColumnTransformer(transformers=[('num',scaler,numerical_columns),('cap', robust_scaler, capital_columns),('cat',hot_encode,categorical_columns)])
X_train_processed=column_transf.fit_transform(X_train)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score





#vamos começar testasndo nosso modelo com 10 vizinhos

model=KNeighborsClassifier(n_neighbors=10)





score=cross_val_score(model,X_train_processed,y_train,cv=5,scoring='accuracy')
score.mean()
folds=np.sqrt(len(X_train)*0.2)

folds
vizinhos=[1,20,40,60,80]

vizinhos_acuracia={}

for i in vizinhos:

    knn=KNeighborsClassifier(n_neighbors=i)

    score=cross_val_score(knn,X_train_processed,y_train,cv=5,scoring='accuracy')

    vizinhos_acuracia[i]=score.mean()

print(vizinhos_acuracia)
melhor_k=max(vizinhos_acuracia,key=vizinhos_acuracia.get)

print("O melhor número de vizinhos foi", melhor_k, "com uma acurácia de",vizinhos_acuracia[melhor_k])
vizinhos2=[10,15,25,30]

vizinhos2_acuracia={}

for i in vizinhos2:

    knn=KNeighborsClassifier(n_neighbors=i)

    score=cross_val_score(knn,X_train_processed,y_train,cv=5,scoring='accuracy')

    vizinhos2_acuracia[i]=score.mean()

print(vizinhos2_acuracia)
vizinhos3=[26,27,28,29]

vizinhos3_acuracia={}

for i in vizinhos3:

    knn=KNeighborsClassifier(n_neighbors=i)

    score=cross_val_score(knn,X_train_processed,y_train,cv=5,scoring='accuracy')

    vizinhos3_acuracia[i]=score.mean()

print(vizinhos3_acuracia)
test_data=pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=["Id"],na_values='?' )
for column in ['workclass', 'native.country', 'occupation']:

    missing_data=test_data[column].mode()[0]

    test_data[column]= test_data[column].fillna(missing_data)
test_data_new=test_data.drop(['education','fnlwgt','native.country'], axis='columns')
X_test=column_transf.fit_transform(test_data_new)
y_train_corect=y_train.replace({0: "<=50K", 1: ">50K"})
KNN=KNeighborsClassifier(n_neighbors=26)

KNN.fit(X_train_processed,y_train_corect)
predicted=KNN.predict(X_test)
predicted
submission=pd.DataFrame()
submission[0]=test_data.index

submission[1]=predicted

submission.columns=['Id','income']
submission.to_csv('finalsubmission.csv',index=False)