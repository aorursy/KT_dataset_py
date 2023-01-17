# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from pandas.tools.plotting import scatter_matrix

from pandas.plotting import autocorrelation_plot



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from mpl_toolkits.mplot3d import axes3d, Axes3D

import seaborn as sns



from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit



from sklearn.svm import SVC

from sklearn.neighbors import NearestCentroid

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import metrics



from itertools import product



import warnings

warnings.filterwarnings('ignore')





# Any results you write to the current directory are saved as output.
# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv("../input/2019-pr-midterm-musicclassification/data_train.csv")

df_data.drop(["filename"], axis=1, inplace=True)
# 전체 데이터를 훈련 / 검증 데이터로 나누는 인덱스를 만든다.

 

split = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state = 42)

for train_index, cross_index in split.split(df_data, df_data["label"]):

    strat_train_set = df_data.iloc[train_index]

    strat_cross_set = df_data.iloc[cross_index]





music = strat_train_set.copy()



music = strat_train_set.drop("label", axis=1)

music_labels = strat_train_set["label"].copy()



music = scale(music);





split_cross = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state = 42)

for test_index, valid_index in split_cross.split(df_data, df_data["label"]):

    strat_test_set = df_data.iloc[test_index]

    strat_valid_set = df_data.iloc[valid_index]





music_valid = strat_valid_set.drop("label", axis=1)

music_valid_labels = strat_valid_set["label"].copy()



music_valid = scale(music_valid);

param_grid = [{'kernel':["poly"], 'gamma': [0.01, 0.1, 0.5], 'C': [0.001, 0.01, 1, 5]}]

svm_poly = SVC()

grid_search = GridSearchCV(svm_poly, param_grid)

grid_search.fit(music_valid, music_valid_labels)
# Load datasets

# DataFrame 을 이용하면 편리하다.

df_data = pd.read_csv('../input/2019-pr-midterm-musicclassification/data_test.csv')

df_data.drop(["filename"], axis=1, inplace=True)







X_test = strat_test_set.drop("label", axis=1)

y_test = strat_test_set["label"].copy()

X_test_prepared = scale(X_test)
clf = grid_search.best_estimator_



y_expected = music_labels

y_predic = clf.predict(music)



score = metrics.accuracy_score(y_expected, y_predic)

print(score)

# numpy 를 Pandas 이용하여 결과 파일로 저장





import pandas as pd



y_expected=np.array(y_expected)

print(y_expected.shape)

df = pd.DataFrame([y_expected])

df = df.replace('blues',0)

df = df.replace('classical',1)

df = df.replace('dog',2)

df = df.replace('country',3)

df = df.replace('disco',4)

df = df.replace('hiphop',5)

df = df.replace('jazz',6)

df = df.replace('metal',7)

df = df.replace('pop',8)

df = df.replace('reggae',9)





df.to_csv('results-yk-v2.csv',index=True, header=False)