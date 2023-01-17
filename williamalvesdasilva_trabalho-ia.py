import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df =  pd.read_csv('../input/Pokemon.csv')
df.head(10)
df.columns = df.columns.str.upper().str.replace('_', '') 
full_data = df
df.head()
df['TYPE 1'].value_counts()
df.info()
df.describe()
df = df.set_index('NAME')
df.head()
df.index = df.index.str.replace(".*(?=Mega)", "")
df.head(10)

df=df.drop(['#'],axis=1)
df['TYPE 2'].fillna(df['TYPE 1'], inplace=True)
df[((df['TYPE 1']=='Fire') | (df['TYPE 1']=='Dragon')) & ((df['TYPE 2']=='Dragon') | (df['TYPE 2']=='Fire'))].head(3)
print("MAx DEFENSE:",df['DEFENSE'].argmax())
print("MAx DEFENSE:",(df['DEFENSE']).idxmax())
print(df.iloc[224])
df.sort_values(by=['DEFENSE'], ascending=False).head(3)


bins=range(0,200,20) 
plt.hist(df["ATTACK"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff') 

# Eixos
plt.xlabel('ATAQUE')
plt.ylabel('QTD')
plt.plot()

# Pontilhado para a média
plt.axvline(df['ATTACK'].mean(),linestyle='dashed',color='red') 
plt.show()
# Separação dos conjuntos
fire=df[(df['TYPE 1']=='Fire') | ((df['TYPE 2'])=="Fire")]
water=df[(df['TYPE 1']=='Water') | ((df['TYPE 2'])=="Water")]

plt.scatter(fire.ATTACK.head(50),fire.DEFENSE.head(50),color='R',label='Fire',marker="*",s=50)
plt.scatter(water.ATTACK.head(50),water.DEFENSE.head(50),color='B',label="Water",s=25)

# Eixos
plt.xlabel("ATAQUE")
plt.ylabel("DEFESA")

plt.legend()
plt.plot()
fig=plt.gcf()  

# Tamanho do quadro
fig.set_size_inches(20,10) 
plt.show()
strong=df.sort_values(by='TOTAL', ascending=False) 
strong.drop_duplicates(subset=['TYPE 1'],keep='first')
df2 = df.drop(['GENERATION','TOTAL'],axis=1)
sns.boxplot(data=df2)
plt.ylim(0,250) 
plt.show()
plt.subplots(figsize = (20,5))
plt.title('Ataque por type 1')
sns.boxplot(x = "TYPE 1", y = "ATTACK",data = df)
plt.ylim(0,200)
plt.show()
plt.subplots(figsize = (20,5))
plt.title('Ataque por type 2')
sns.boxplot(x = "TYPE 2", y = "ATTACK",data=df)
plt.show()
plt.subplots(figsize = (15,5))
plt.title('Defesa por Type 1')
sns.boxplot(x = "TYPE 1", y = "DEFENSE",data = df)
plt.show()
plt.figure(figsize=(20,10))
top_types=df['TYPE 1'].value_counts()
df1=df[df['TYPE 1'].isin(top_types.index)] 
sns.swarmplot(x='TYPE 1',y='TOTAL',data=df1,hue='LEGENDARY')
plt.axhline(df1['TOTAL'].mean(),color='red',linestyle='dashed')
plt.show()
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
plt.show()
datax = df.drop(["TYPE 1","TYPE 2","GENERATION"],axis=1)
_ =sns.pairplot(datax, hue='LEGENDARY', diag_kind='kde', height=2)
_ = 10
data = full_data.drop(['TYPE 2'],axis='columns')
data.LEGENDARY.value_counts()
lengendary = data.loc[data['LEGENDARY']==True]
lengendary = lengendary.append(lengendary.append(lengendary))
lengendary.LEGENDARY.value_counts()
full_data = data.append(lengendary.append(lengendary.append(lengendary)))
full_data
full_data['LEGENDARY'] = full_data.LEGENDARY.map({False: 0, True: 1})
full_data
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
xyz = make_column_transformer(
            (OneHotEncoder(),['TYPE 1','GENERATION']),
            (StandardScaler(),['TOTAL','HP','ATTACK','DEFENSE','SP. ATK','SP. DEF','SPEED']), remainder = 'passthrough')
full_data = full_data.drop(['#','NAME'],axis='columns')
X = full_data.drop(['LEGENDARY'], axis = 1)
y = full_data['LEGENDARY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print(len(X), len(y), len(X_train), len(X_test), len(y_train), len(y_test))
xyz.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
logreg = LogisticRegression(solver='lbfgs')
pipe = make_pipeline(xyz,logreg)
from sklearn.model_selection import cross_val_score
from sklearn import metrics

print('Score do dataset de Treino: {}'.format(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()*100))

pipe = make_pipeline(xyz,logreg)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print('Score do dataset de Teste: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.svm import SVC

svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    pipe = make_pipeline(xyz,svc_classifier)
    svc_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
from matplotlib.cm import rainbow

colors = rainbow(np.linspace(0, 1, len(kernels)))
plt.figure(figsize=(10,7))
plt.bar(kernels, svc_scores, color = colors)

for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i])
    
plt.xlabel('Kernels')
plt.ylabel('Scores')

print('Score do dataset de Treino: {}'.format(svc_scores[0]*100))

svc_classifier = SVC(kernel = 'linear')
pipe = make_pipeline(xyz, svc_classifier)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print('Score do dataset de Teste:: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.ensemble import RandomForestClassifier

rf_scores = []
estimators = [10, 100, 200, 500, 1000]

for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    pipe = make_pipeline(xyz, rf_classifier)
    rf_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(10,7))
colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)

for i in range(len(estimators)):
    plt.text(i, rf_scores[i], round(rf_scores[i],5))
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])

plt.xlabel('Estimadores')
plt.ylabel('Score')

print('Score do dataset de Treino: {}'.format(rf_scores[0]*100))

rf_classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)
pipe = make_pipeline(xyz,rf_classifier)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print('Score do dataset de Teste: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))