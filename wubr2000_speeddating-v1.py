# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import scipy as sp
import random 
from sklearn.ensemble import RandomForestClassifier as RF
import operator
import time

df = pd.read_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1")
df.shape
df.describe()
#Only consider these candidates:
#Condition = extensive choice
#Goal = NOT "seemed like a fun night out" and NOT "to say I did it"
extensive_df = df[(df.condtn==2) & ((df.goal != 1) & (df.goal != 5))]
extensive_df.shape
extensive_df.head(5)
important_features = \
[
    'order',
    'int_corr',
    'samerace',
    'age_o',
#    'race_o', need to dummify
    'pf_o_att',
    'attr_o',
    'field_cd',
#    'undergra', has alot of NA
#    'mn_sat', has alot of NA
#    'tuition', has alot of NA
#    'race', need to dummify
    'imprace',
    'imprelig',
#    'from',
#    'income',
    'date',
    'go_out',
    'career_c',
    'sports',
    'tvsports',
    'exercise',
    'dining',
    'museums',
    'art',
    'hiking',
    'gaming',
    'clubbing',
    'reading',
    'tv',
    'theater',
    'movies',
    'concerts',
    'music',
    'shopping',
    'yoga',
    'exphappy',
    'expnum'
]
Y = extensive_df['dec_o']
X = extensive_df[important_features]
#Concatenate all X and Y
df_final = pd.concat([Y, X], axis=1).reset_index(drop=True)
df_final = df_final.replace([np.inf, -np.inf], np.nan)
df_final = df_final.fillna(0)
df_final.isnull().values.any()
df_final_np = df_final.values
#UP-SAMPLING APPROACH - OVER-SAMPLE THE MINORITY CLASS
matched_rows = np.where(df_final_np[:,0]==1.0)[0]
nonmatched_rows = np.where(df_final_np[:,0]==0.0)[0]

nonmatched_percentage = 0.50
nonmatched_sample_size = int(df_final_np.shape[0] * nonmatched_percentage)
matched_sample_size = int(df_final_np.shape[0] * (1 - nonmatched_percentage))

sampled_nonmatched_rows = np.random.choice(nonmatched_rows, nonmatched_sample_size, replace=False)
sampled_matched_rows = np.random.choice(matched_rows, matched_sample_size, replace=True)

df_upsampled_final = np.concatenate((df_final_np[sampled_nonmatched_rows], df_final_np[sampled_matched_rows]), axis=0)
X_final_np = df_upsampled_final[:,1:]
Y_final_np = df_upsampled_final[:,0]
df_upsampled_final.shape
X_final_np.shape
Y_final_np.shape
#Setting up K-fold CV function:
from sklearn.cross_validation import KFold, StratifiedKFold 
def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    # kf = KFold(len(y),n_folds=4,shuffle=True)
    #Stretified KFold preserves the class balance in the training and test samples
    #Better for when there's a class imbalance and after up/down-sampling
    kf = StratifiedKFold(y, n_folds=2)
    
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred, clf
def accuracy(y_true,y_pred):
#     NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)
print ("Random forest:")
y_pred_RF, rf_model = run_cv(X_final_np, Y_final_np , RF,
                    criterion = "entropy",
                    n_estimators = 200,
                    max_features = 'sqrt',
                    oob_score = True,
                    max_depth = 2,
                    n_jobs = -1,
                    verbose = 1
                   )

print ("Accuracy: %.3f" % accuracy(Y_final_np, y_pred_RF))
#Feature importance
rf_dict = dict((k, v) for k, v in dict(zip(df_final.columns[1:],rf_model.feature_importances_)).items())
sorted(rf_dict.items(), key=operator.itemgetter(1), reverse=True)
