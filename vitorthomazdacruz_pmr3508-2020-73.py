import pandas as pd # biblioteca utilizada para a leitura e mannipulação do database
df = pd.read_csv('train_data.csv',
    names = ['Id','Age','Workclass','fnlwgt','Education','Education Number',
             'Marital Status','Occupation','Relationship','Race','Sex',
             'Capital Gain','Capital Loss','Hours per Week','Native Country',
             'Income'], # feature que queremos estimar
    na_values='?')
df.head() # aqui vemos que há uma linha com os nomes das features, a ser descartada posteriormente
df.tail()
df.drop(0,inplace=True) # descarte da primeira linha
df
df.duplicated().value_counts()
df.describe()
#from pandas_profiling import ProfileReport
#ProfileReport(df,title='Profiling Report of Adult Database')
df.drop(['Education'],axis=1,inplace=True)
df.head()
import seaborn as sns # biblioteca utilizada para gerar os gráficos
sns.distplot(df['Age'])
df['Age'].isnull().sum() # não há valores faltando
sns.countplot(y='Workclass',data=df,order=df['Workclass'].value_counts().index)
df['Workclass'].value_counts()
df['Workclass'].isnull().sum()
sns.distplot(df['fnlwgt'])
df['fnlwgt'].isnull().sum()
df.drop('fnlwgt',axis=1,inplace=True)
sns.countplot(df['Education Number'],order=df['Education Number'].value_counts(ascending=True).index)
df['Education Number'].value_counts()
sns.countplot(y='Marital Status',data=df,order=df['Marital Status'].value_counts().index)
df['Marital Status'].value_counts()
sns.countplot(y='Occupation',data=df,order=df['Occupation'].value_counts().index)
df['Occupation'].value_counts()
sns.countplot(y='Relationship',data=df,order=df['Relationship'].value_counts().index)
df['Relationship'].value_counts()
sns.countplot(y='Race',data=df,order=df['Race'].value_counts().index)
df['Race'].value_counts()
sns.countplot(y='Sex',data=df,order=df['Sex'].value_counts().index)
df['Sex'].value_counts()
sns.distplot(df['Capital Gain'])
df['Capital Gain'].value_counts()
sns.distplot(df['Capital Loss'])
df['Capital Loss'].value_counts()
#df.drop(['Capital Gain','Capital Loss'],axis=1,inplace=True)
sns.distplot(df['Hours per Week'])
df['Hours per Week'].value_counts()
df['Native Country'].value_counts()
#df.drop(['Native Country'],axis=1,inplace=True)
df
features = ['Id','Age','Education Number',
             'Marital Status','Relationship','Race','Sex',
             'Capital Gain','Capital Loss','Hours per Week',
             'Income']
train = df[features]
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in train.columns:
    train[i] = label_encoder.fit_transform(train[i])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
y = train.pop('Income')
x = train
x = scaler.fit_transform(x)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
score = cross_val_score(knn,x,y,cv=10, scoring="accuracy")
print('Acurácia com cross validation:',score.mean())
tmp = 0
for i in range(30):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn,x,y,cv=10, scoring="accuracy")
    if score.mean() > tmp:
        tmp = score.mean()
        n = i
    print(f'Acurária para {i} vizinhos: {score.mean()}')
test = pd.read_csv('test_data.csv',
           names = ['Id','Age','Workclass','fnlwgt','Education','Education Number',
             'Marital Status','Occupation','Relationship','Race','Sex',
             'Capital Gain','Capital Loss','Hours per Week','Native Country'],
           sep=',',engine='python',na_values='?')
test.drop(0,inplace=True)
test.head()
x_test = test[['Id','Age','Education Number',
             'Marital Status','Relationship','Race','Sex',
             'Capital Gain','Capital Loss','Hours per Week',]]
knn = KNeighborsClassifier(n_neighbors=n)
score = cross_val_score(knn,x,y,cv=10, scoring="accuracy")
print('Acurácia com cross validation:',score.mean())
for i in x_test.columns:
    x_test[i] = label_encoder.fit_transform(x_test[i])
x_test
knn.fit(x,y)
y_pred = knn.predict(x_test)
import numpy as np
result = np.vstack((test['Id'],y_pred)).T
x = ["id","income"]
submit = pd.DataFrame(columns = x, data = result)
submit.to_csv("Resultados.csv", index = False)

