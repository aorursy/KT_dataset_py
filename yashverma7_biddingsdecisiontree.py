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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
df= pd.read_csv('../input/rtb/biddings.csv')
print(df.shape)
count_classes = pd.value_counts(df['convert'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("bidding conversion histogram")
plt.xlabel("Conversion")
plt.ylabel("Count")
train = df[:800000]
test = df[800000:]
def undersample(df, ratio=1):
    conv = df[df.convert == 1]
    oth = df[df.convert == 0].sample(n=ratio*len(conv))
    return pd.concat([conv, oth]).sample(frac=1) #shuffle data

ustrain = undersample(train)

y = ustrain.convert
x = ustrain.drop('convert', axis=1)

print("Remaining rows", len(ustrain))
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import auc,roc_curve
from sklearn.metrics import f1_score
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print('Shape of X_train=>',X_train.shape)
print('Shape of X_test=>',X_test.shape)
print('Shape of Y_train=>',Y_train.shape)
print('Shape of Y_test=>',Y_test.shape)
# Building Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train, Y_train)
dt_pred_train = dt.predict(X_train)
# Evaluation on Training set
dt_pred_train = dt.predict(X_train)
print('Training Set Evaluation F1-Score=>',f1_score(Y_train,dt_pred_train))
# Evaluating on Test set
dt_pred_test = dt.predict(X_test)
print('Testing Set Evaluation F1-Score=>',f1_score(Y_test,dt_pred_test))
from sklearn.tree import DecisionTreeClassifier
msl_s = [1,2,4,8,16,32,64,128,256]
scores = list()
scores_std = list()
dect = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                              random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)

for msl in msl_s:
    dect.min_samples_leaf = msl
    this_scores = cross_val_score(dect, x, y, cv=4,scoring='roc_auc')
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
    
dect_results = pd.DataFrame({'score':scores, 'Minimum samples leaf': msl_s}) 
dect_results
y_preds = []
dect.min_samples_leaf = int(dect_results.loc[dect_results['score'].idxmax()]['Minimum samples leaf'])
y_preds.append(dect.fit(x,y).predict_proba(test.drop('convert', axis=1))[:,1])
model = ['Decision Tree']
colors = ['b']

for i in range(0,1):
    fpr, tpr, thresholds = roc_curve(test.convert,y_preds[i])
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, 'b',label='%s AUC = %0.2f'% (model[i] ,roc_auc),  color=colors[i], linestyle='--')
    plt.legend(loc='lower right')
    
plt.title('Receiver Operating Characteristic')
plt.plot([-0.1,1.1],[-0.1,1.1],color='gray', linestyle=':')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()