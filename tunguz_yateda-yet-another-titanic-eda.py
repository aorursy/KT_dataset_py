# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost
import shap
import numpy as np
import matplotlib.pylab as pl

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
PassengerId = test['PassengerId']
train.head()
train.shape

test.shape
train.info()
test.info()
train.isnull().values.any()
test.isnull().values.any()
train.describe()
test.describe()
target = train['Survived']
features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_len = train.shape[0]
train_len
train_test = train[features_to_use].append(test[features_to_use])
train_test.shape
le = LabelEncoder()
train_test['Sex'] = le.fit_transform(train_test['Sex'].values)
train_test.head()
train_test.loc[(train_test['Embarked'].isnull()), 'Embarked']='S'
le = LabelEncoder()
train_test['Embarked'] = le.fit_transform(train_test['Embarked'].values)
train_test.head()
train = train_test[:train_len]
test = train_test[train_len:]
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)
params = {
    "eta": 0.001,
    "objective": "binary:logistic",
    "base_score": np.mean(y_train),
    "eval_metric": "logloss",
    'silent': True
}
model = xgboost.train(params, d_train, 10000, evals = [(d_test, "test")], early_stopping_rounds=50, verbose_eval=False)
xgboost.plot_importance(model)
pl.title("xgboost.plot_importance(model)")
pl.show()
xgboost.plot_importance(model, importance_type="cover")
pl.title('xgboost.plot_importance(model, importance_type="cover")')
pl.show()
xgboost.plot_importance(model, importance_type="gain")
pl.title('xgboost.plot_importance(model, importance_type="gain")')
pl.show()
shap_values = shap.TreeExplainer(model).shap_values(train)
shap.initjs()
shap.force_plot(shap_values[0,:], train.iloc[0,:])
shap.force_plot(shap_values, train)
global_shap_vals = np.abs(shap_values).mean(0)[:-1]
inds = np.argsort(global_shap_vals)
y_pos = np.arange(train.shape[1])
pl.barh(y_pos, global_shap_vals[inds], color="#1E88E5")
pl.yticks(y_pos, train.columns[inds])
pl.gca().spines['right'].set_visible(False)
pl.gca().spines['top'].set_visible(False)
pl.xlabel("mean SHAP value magnitude (change in log odds)")
pl.gcf().set_size_inches(6, 4.5)
pl.show()
shap.summary_plot(shap_values, train)
for name in X_train.columns:
    shap.dependence_plot(name, shap_values, train)
shap_pca50 = PCA(n_components=3).fit_transform(shap_values)
shap_embedded = TSNE(n_components=2, perplexity=50).fit_transform(shap_values)
cdict1 = {
    'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
            (1.0, 0.9607843137254902, 0.9607843137254902)),

    'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
              (1.0, 0.15294117647058825, 0.15294117647058825)),

    'blue': ((0.0, 0.8980392156862745, 0.8980392156862745),
             (1.0, 0.3411764705882353, 0.3411764705882353)),

    'alpha': ((0.0, 1, 1),
              (0.5, 1, 1),
              (1.0, 1, 1))
}  # #1E88E5 -> #ff0052
red_blue_solid = LinearSegmentedColormap('RedBlue', cdict1)
f = pl.figure(figsize=(5,5))
pl.scatter(shap_embedded[:,0],
           shap_embedded[:,1],
           c=shap_values.sum(1).astype(np.float64),
           linewidth=0, alpha=1., cmap=red_blue_solid)
cb = pl.colorbar(label="Log odds of surviving", aspect=40, orientation="horizontal")
cb.set_alpha(1)
cb.draw_all()
cb.outline.set_linewidth(0)
cb.ax.tick_params('x', length=0)
cb.ax.xaxis.set_label_position('top')
pl.gca().axis("off")
pl.show()
for feature in features_to_use:
    f = pl.figure(figsize=(5,5))
    pl.scatter(shap_embedded[:,0],
               shap_embedded[:,1],
               c=train[feature].values.astype(np.float64),
               linewidth=0, alpha=1., cmap=red_blue_solid)
    cb = pl.colorbar(label=feature, aspect=40, orientation="horizontal")
    cb.set_alpha(1)
    cb.draw_all()
    cb.outline.set_linewidth(0)
    cb.ax.tick_params('x', length=0)
    cb.ax.xaxis.set_label_position('top')
    pl.gca().axis("off")
    pl.show()
preds = model.predict(xgboost.DMatrix(test))
preds = np.round(preds).astype('int')
submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': preds})
submission.head()
submission.to_csv('submission.csv', index=False)
