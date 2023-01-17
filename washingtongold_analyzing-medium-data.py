# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_excel('/kaggle/input/medium-views-dataset/Medium data.xlsx')
data.head()
def add(string):

    try:

        return float(string.split(',')[0])+float(string.split(',')[1])

    except:

        return string

data['Complete Money'] = data['Money Earned'].apply(add)
from datetime import datetime

def difference(date):

    return (datetime(2020,4,8) - date).days

data['Age'] = data['Publish Date'].apply(difference)
data.head()
data.columns
#data=data[data['Title']!="Why You Shouldn't Believe the Coronavirus Death Rate"]
from sklearn.model_selection import train_test_split

X = data[['Reading Time','Views','Reads','Fans','Published?','Age']]

y = data['Complete Money']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

dtr.fit(X_train,y_train)
from sklearn.linear_model import LinearRegression

lin = LinearRegression()

lin.fit(X,y)
from sklearn.metrics import mean_absolute_error as MAE

MAE(dtr.predict(X_test),y_test)
MAE(lin.predict(X_test),y_test)
coef = pd.DataFrame({'Column':X_test.columns,'Coefficient':lin.coef_}).sort_values('Coefficient')

coef
import shap

#e#xplainer = shap.TreeExplainer()

#shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type="bar")
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(dtr, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
data.columns
from pdpbox import pdp, info_plots

import matplotlib.pyplot as plt



base_features = ['Reading Time','Views','Reads','Fans','Published?','Age']

for feat_name in base_features:

    #feat_name = 'Fans'

    pdp_dist = pdp.pdp_isolate(model=dtr, dataset=X_test, model_features=base_features, feature=feat_name)



    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()
for feature1 in base_features:

    for feature2 in base_features:

        try:

            features = [feature1,feature2]

            inter1  =  pdp.pdp_interact(model=dtr, dataset=X_test, model_features=base_features, 

                                        features=features)

            pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features, plot_type='contour')

            plt.show()

        except:

            pass
from sklearn.linear_model import LinearRegression

lin = LinearRegression()

lin.fit(X_train,y_train)

coef = pd.DataFrame({'Column':X_train.columns,'Coefficient':lin.coef_}).sort_values('Coefficient')
coef
import seaborn as sns

sns.barplot(coef['Column'],coef['Coefficient'])
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(max_features=50,stop_words='english')

vocab = count.fit_transform(data['Title'])
from sklearn.feature_extraction.text import CountVectorizer

count2 = CountVectorizer(max_features=50,stop_words='english')

count2.fit(data['Subtitle'])

vocab2 = count.fit_transform(data['Subtitle'])
from tqdm import tqdm

words = []

for i in tqdm(range(50)):

    clean_list = [0 for i in range(50)]

    clean_list[i] = 1

    words.append(count.inverse_transform(clean_list)[0][0])
from tqdm import tqdm

words2 = []

for i in tqdm(range(50)):

    clean_list2 = [0 for i in range(50)]

    clean_list2[i] = 1

    words2.append(count2.inverse_transform(clean_list2)[0][0])
vocab = vocab.todense()
vocab2 = vocab2.todense()
keys = list(count.vocabulary_.keys())
keys2 = list(count2.vocabulary_.keys())
for i in range(50):

    data[str('Title Word '+str(i)+':'+keys[i])] = np.array(vocab)[:,i].reshape((1,-1))[0]
for i in range(50):

    data[str('Subtitle Word '+str(i)+':'+keys2[i])] = np.array(vocab2)[:,i].reshape((1,-1))[0]
data.columns
X_1 = data.drop(['Title', 'Subtitle', 'Publish Date', 'Reading Time', 'Money Earned',

       'Views', 'Reads', 'Fans', 'Published?', 'Complete Money', 'Age'],axis=1)

y_1 = data['Complete Money']
X_train,X_test,y_train,y_test = train_test_split(X_1,y_1)
dec = DecisionTreeRegressor()

dec.fit(X_train,y_train)
from sklearn.ensemble import RandomForestRegressor

ran = RandomForestRegressor()

ran.fit(X_train,y_train)
lin = LinearRegression()

lin.fit(X_train,y_train)
MAE(dec.predict(X_test),y_test)
MAE(lin.predict(X_test),y_test)
data
explainer = shap.TreeExplainer(ran)

values = explainer.shap_values(X_1)
shap.summary_plot(values, X_1,plot_type='bar')
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(ran, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(dec, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
log = LinearRegression()

log.fit(X_train,y_train)
(log.predict(X_test) - y_test).apply(abs).mean()
words = pd.DataFrame({'Word':X_test.columns,'Coefficient':log.coef_}).sort_values('Coefficient')
words = words.reset_index().drop('index',axis=1)
key1 = list(count.vocabulary_.keys())

key2 = list(count2.vocabulary_.keys())
key1.extend(key2)
total_keys = keys

for item in keys2:

    total_keys.append(item)
keys2
import matplotlib.pyplot as plt

plt.figure(figsize=(3,30))

sns.heatmap(np.array([np.array(words['Coefficient'])]).T,annot=True,yticklabels=key1)

plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(3,30))

sns.heatmap(np.array([np.array(words['Coefficient'])]).T,annot=True,yticklabels=totalkeys)

plt.show()
data.head()