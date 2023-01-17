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
!pip install pycaret
!pip install shap 
from pycaret.classification import *
df = pd.read_csv('../input/predictive-modeling-training/training.csv', sep= ';')
df.head()
clf1 = setup(df, target = 'left', session_id=123, log_experiment=False, 
             numeric_features = ['number_project','time_spend_company'],
            ordinal_features= {'salary' : ['low','medium','high'] }, normalize=True, feature_interaction=True, feature_selection=True)
compare_models()
cat = create_model('catboost', auto_class_weights='Balanced')
rf = create_model('rf', class_weight = 'balanced')
tuned_rf = tune_model(rf, optimize='AUC')
rf.get_params()
tuned_rf.get_params()
tuned_cat = tune_model(cat, fold=4, optimize='AUC')
tuned_cat.get_all_params()
cat.get_all_params()
test = pd.read_csv('../input/predictive-modeling-training/testing.csv', sep=';')
test.head()
predict_new = predict_model(tuned_rf, data=test)
predict_new.head()
final = predict_new[['id','Label']]
final.columns = ['id', 'left']
final.head(20)
final.to_csv('tuned_rf.csv',index=False)
from catboost import Pool
pool = Pool(clf1[0], clf1[1], feature_names=list(clf1[0].columns))
cat.get_feature_importance(type='ShapInteractionValues',data=pool)
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(cat, clf1[3], clf1[5],
                                 display_labels=['N','Y'],
                                 cmap=plt.cm.Greens,
                                 normalize=None)
interpret_model(cat)
interpret_model(tuned_cat, plot='correlation', feature='satisfaction_level')
interpret_model(tuned_cat, plot='correlation', feature='time_spend_company')
interpret_model(tuned_cat, plot='correlation', feature='number_project')
interpret_model(tuned_cat, plot='correlation', feature='average_montly_hours')
interpret_model(tuned_cat, plot='correlation', feature='last_evaluation')
interpret_model(tuned_cat, plot='correlation', feature='salary')
interpret_model(tuned_cat )