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
SEED = 15

train = pd.read_csv('../input/airline-passenger-satisfaction/train.csv')

test = pd.read_csv('../input/airline-passenger-satisfaction/test.csv')
train.head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn import tree
#clears the dataset of undefined values

def clean_dataset(data):

    assert isinstance(data, pd.DataFrame), "df needs to be a pd.DataFrame"

    data.dropna(inplace=True)

    indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
def find_cat(data, max_count_unique=5):

    for name in data.columns:

        s = ''

        s += name

        if type(data[name][0]) == str:

            s += ' is string, '

        if data[name].nunique() <= max_count_unique:

            s += ' few unique'

        if s != name:

            print(s, data[name].unique())



#replacing categorical variables

def encoding_cat(data, max_count_unique=5, msg=True):

    for name in data.columns:

        if type(data[name][0]) == str and data[name].nunique() <= max_count_unique:

            le = LabelEncoder()

            le.fit(data[name])

            data[name] = le.transform(data[name])

    if msg:

        print('Encoding done!')

            
clean_dataset(train)

find_cat(train)

encoding_cat(train)
train.info()
train.Class = train.Class.replace({0: 3}) 

#Correcting Class variable in accordance with the meaning Eco -> Eco Plus -> Business

train['Arrival Delay in Minutes'].astype('int')

y = train.satisfaction

X = train.drop(['Unnamed: 0', 'id', 'satisfaction'], axis=1)

X.head()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 10, random_state=SEED)

decision_tree.fit(X_train, y_train)
from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw

import graphviz  

from sklearn.tree import export_graphviz



# Export our trained model as a .dot file

with open("tree1.dot", 'w') as f:

     f = export_graphviz(decision_tree, out_file=f, max_depth = 4,

                         impurity = True, feature_names = X_train.columns,

                         rounded = True, filled= True )

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree.png'])

# Annotating chart with PIL

img = Image.open("tree.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
decision_tree.score(X_val, y_val)
def search_param(model, param, X_train, y_train, X_val, y_val, area=range(1, 11), msg=True, plot=True, seed=None):

    import matplotlib.pyplot as plt

    import time

    score_list = []

    if msg:

        print('#     accuracy  time')

    for i in area:

        start = time.time()

        rfc = eval(model + '(' + param + '=' + str(i) + ', random_state=' + str(seed) + ')')

        rfc.fit(X_train, y_train)

        s = rfc.score(X_val, y_val)

        end = time.time()

        score_list.append(s)

        if msg:

            print("%-3d %10f  %7f" % (i, s, end - start))

    if plot:

        plt.plot(list(area), score_list)

    return list(area)[score_list.index(max(score_list))]
search_param('RandomForestClassifier', 'n_estimators', X_train, y_train, X_val, y_val, area=range(1, 51), seed=SEED)
search_param('RandomForestClassifier', 'max_depth', X_train, y_train, X_val, y_val, range(1, 25), seed=SEED)
search_param('RandomForestClassifier', 'min_samples_split', X_train, y_train, X_val, y_val, range(2, 10), seed=SEED)
search_param('RandomForestClassifier', 'min_samples_leaf', X_train, y_train, X_val, y_val, range(1, 10), seed=SEED)
rfc = RandomForestClassifier(random_state=SEED)

param = {'n_estimators': [i for i in range(38, 51)], 'max_depth': [i for i in range(20, 25)]}

gscv =  GridSearchCV(rfc, param, cv=3, n_jobs=-1, verbose=1)

gscv.fit(X_train, y_train)
gscv.best_params_
best_c = gscv.best_estimator_

imp = pd.DataFrame(best_c.feature_importances_, index=X_train.columns, columns=['importance'])

imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
best_c.score(X_val, y_val)
clean_dataset(test)

find_cat(test)

encoding_cat(test)

test.Class = test.Class.replace({0: 3})

test['Arrival Delay in Minutes'].astype('int')

y_test = test['satisfaction']

X_test = test.drop(['Unnamed: 0', 'id', 'satisfaction'], axis=1)

best_c.score(X_test, y_test)
from sklearn.metrics import roc_auc_score , roc_curve

import matplotlib.pyplot as plt

dtc_proba=best_c.predict_proba(X_test)

dtc_proba=dtc_proba[:,1]

auc=roc_auc_score(y_test, dtc_proba)

print('Random Forest Classifier: ROC AUC=%.3f' % (auc))

lr_fpr, lr_tpr, _ = roc_curve(y_test, dtc_proba)

plt.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest Classifier')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
!rm -rf ./sample-out.png ./tree1.dot ./tree.png