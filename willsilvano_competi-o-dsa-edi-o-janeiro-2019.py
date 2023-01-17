import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import itertools



from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from xgboost import plot_importance



from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer, Binarizer, KernelCenterer, MaxAbsScaler, MinMaxScaler, RobustScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline



from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import precision_recall_curve



import warnings

warnings.filterwarnings('ignore')
df_treino = pd.read_csv('../input/dataset_treino.csv')

df_teste = pd.read_csv('../input/dataset_teste.csv')
df_treino.head(5)
df_teste.head(5)
print("O dataset treino tem {} linhas e {} colunas".format(df_treino.shape[0], df_treino.shape[1]))

print("O dataset teste tem {} linhas e {} colunas".format(df_teste.shape[0], df_teste.shape[1]))
df_treino.dtypes
df_teste.dtypes
df_treino.drop(columns='id', inplace=True)
target = 'classe'

features = df_treino.columns.tolist()

features.remove(target)



X = df_treino[features]

y = df_treino[target]



print('Variável alvo:', target)

print('Variável preditoras:', features)
df_treino.isnull().sum()
df_teste.isnull().sum()
for var in ['glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi']:

    df_treino[var].replace(0, np.NAN, inplace=True)

    df_treino[var].fillna(df_treino[var].mean(), inplace=True)
labels = ['Com diabetes' if label == 1 else 'Sem diabetes' for label in y]

palette = {'Com diabetes': '#f56b69', 'Sem diabetes': '#3bbbb3'}



ax = sns.countplot(labels, palette=palette)

plt.title('Quantidade de pessoas com e sem diabetes')

plt.ylabel('Quantidade de pessoas')

plt.show()



porcentagem_tipos = y.value_counts(normalize=True)*100

contagem_tipos = y.value_counts()

print('Com diabetes: {} ({}%)'.format(contagem_tipos[1], round(porcentagem_tipos[1],2)))

print('Sem diabetes: {} ({}%)'.format(contagem_tipos[0], round(porcentagem_tipos[0],2)))
df_treino.groupby('classe').describe()
i = 1

for feature in features:                       

    if i == 1:

        plt.figure(figsize=[20,5])

                

    plt.subplot(1, 3, i)

    sns.boxplot(y=feature, data=X)

    plt.title('Boxplot da variável {}'.format(feature))

    

    i+=1



    if i == 4:    

        plt.tight_layout()

        plt.show()

        i = 1
Q1 = df_treino.quantile(0.25)

Q3 = df_treino.quantile(0.75)

IQR = Q3 - Q1

print('IQR das variáveis:\n')

print(IQR)



print()

print('Dimensões do dataset com outliers:', df_treino.shape)



df_treino = df_treino[~((df_treino < (Q1 - 1.5 * IQR)) |(df_treino > (Q3 + 1.5 * IQR))).any(axis=1)]

X = df_treino[features]

y = df_treino[target]



print('Dimensões do dataset sem outliers:', df_treino.shape)
i = 1

for feature in features:                       

    if i == 1:

        plt.figure(figsize=[20,5])

                

    plt.subplot(1, 3, i)

    sns.boxplot(y=feature, data=X)

    plt.title('Boxplot da variável {}'.format(feature))

    

    i+=1



    if i == 4:    

        plt.tight_layout()

        plt.show()

        i = 1
sns.pairplot(data=df_treino, hue='classe', diag_kind='kde', palette={1: '#f56b69', 0: '#3bbbb3'})

plt.show()
correlacao = df_treino.corr()

plt.figure(figsize=(14,14))



sns.heatmap(correlacao, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size': 15}, cmap='coolwarm')

plt.show()
model = XGBClassifier()

model.fit(X, y)

plot_importance(model)

plt.show()
# X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.3, random_state=42)

# X_train.head()



X_train = X

y_train = y
pipelines = []



for clf in [

    LogisticRegression(),

    SVC(kernel='linear'),

    KNeighborsClassifier(),

    RandomForestClassifier(),

    XGBClassifier(),

    GaussianNB(),

    MLPClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier()

]:

    

    for normalizer in [

        None,

        StandardScaler(),

        Normalizer(), 

        Binarizer(), 

        FunctionTransformer(np.log1p),

        MaxAbsScaler(), 

        MinMaxScaler(),

    ]:

        clf_pipeline = make_pipeline(normalizer, clf)

        

        pipelines.append({

            'Algoritmo': clf.__class__.__name__,

            'Normalizer': normalizer.__class__.__name__,

            'Acuracia': cross_val_score(clf_pipeline, X_train, y_train, cv=10).mean()

        })

        

pd.DataFrame(pipelines).sort_values(by='Acuracia', ascending=False)
pipeline = Pipeline([

    ('normalizer',MinMaxScaler()),

    ('classifier', MLPClassifier()),

])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(df_teste[features])



df_submission = df_teste[['id']]

df_submission['classe'] = preds

df_submission.to_csv('submission.csv', index=False)