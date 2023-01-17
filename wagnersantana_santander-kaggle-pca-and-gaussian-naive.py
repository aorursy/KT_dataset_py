import pandas as pd
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import folium
%matplotlib inline



import warnings
warnings.filterwarnings("ignore")
%time train_data = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
%time test_data = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
train_data.head()
test_data.head()
train_data.info()
sns.countplot(y=train_data.target ,data=train_data)
plt.xlabel("Contagem dos valores da target")
plt.ylabel("Target")
plt.show()
train_data.hist(figsize=(30,24),bins = 15)
plt.title("Distribuição das variáveis")
plt.show()
from sklearn.preprocessing import MinMaxScaler

mmscale = MinMaxScaler()  
X_train = mmscale.fit_transform(train_data.drop(['ID_code','target'],axis=1))  
X_test = mmscale.transform(test_data.drop(['ID_code'], axis=1))
from sklearn.decomposition import PCA

pca = PCA()  
a = pca.fit_transform(X_train) 
b = pca.transform(X_test)
variancia_pca = pca.explained_variance_ratio_
pd.DataFrame(variancia_pca,columns=['Variância após o PCA']).plot(kind='box')
with plt.style.context('classic'):
    plt.figure(figsize=(12, 9))

    plt.bar(range(200), variancia_pca, alpha=0.5, align='center',
            label='variância individual')
    plt.ylabel('Taxa de variância')
    plt.xlabel('Componentes principais')
    plt.legend(loc='best')
    plt.tight_layout()
sum(variancia_pca[:100])
X_train = train_data.iloc[:, 2:].values.astype('float64')
y_train = train_data['target'].values
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import QuantileTransformer

pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())
pipeline.fit(X_train, y_train)
from sklearn.metrics import roc_curve, auc

fpr, tpr, thr = roc_curve(y_train, pipeline.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('Taxa de falsos positivos')
plt.ylabel('Taxa verdadeiros positivos')
plt.title('Curva ROC')
auc(fpr, tpr)
from sklearn.model_selection import cross_val_score

cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=10).mean()