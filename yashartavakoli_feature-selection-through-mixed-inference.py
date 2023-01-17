import numpy as np 

import pandas as pd 



dataset = pd.read_csv("../input/mushroom-classification/mushrooms.csv")



X = dataset.iloc[:, 1:]

X = X.astype(str)

y = dataset.iloc[:,0]



from sklearn.preprocessing import OrdinalEncoder



oe = OrdinalEncoder()

oe.fit(X)

X_enc = oe.transform(X)



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(y)

y_enc = le.transform(y)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



svc = SVC()

parameters = {'C':(1, 10, 100), 'gamma':(0.1 ,1, 10)}

clf_full_features = GridSearchCV(svc, parameters)

clf_full_features.fit(X_enc, y_enc)

print(clf_full_features.best_score_)
svc = SVC()

parameters = {'C':(5,200,300), 'gamma':(0.01, 0.05,0.1)}

clf_full_features = GridSearchCV(svc, parameters)

clf_full_features.fit(X_enc, y_enc)

print(clf_full_features.best_score_)
svc = SVC()

parameters = {'C':(200,300,400), 'gamma':(0.01, 0.05,0.1)}

clf_full_features = GridSearchCV(svc, parameters)

clf_full_features.fit(X_enc, y_enc)

print(clf_full_features.best_score_)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



fs1 = SelectKBest(score_func=chi2, k='all')

fs1.fit(X_enc, y_enc)

X_fs1 = fs1.transform(X_enc)



from sklearn.feature_selection import mutual_info_classif



fs2 = SelectKBest(score_func=mutual_info_classif, k='all')

fs2.fit(X_enc, y_enc)

X_fs2 = fs2.transform(X_enc)



import matplotlib.pyplot as plt



x = np.arange(len(fs1.scores_))



fig, ax = plt.subplots()

fig.set_size_inches(18.5, 10.5)



width = 0.35  

rects1 = ax.bar(x - width/2, fs1.scores_, width, label='Chi-squared')

rects2 = ax.bar(x + width/2, fs2.scores_*10000, width, label='Mutual Information * 10000')



ax.set_ylabel('Feature Importance',fontsize=20)

ax.set_title('Feature',fontsize=20)

ax.set_xticks(x)

ax2 = ax.twiny()

ax2.set_xlim(ax.get_xlim())

ax2.set_xticks(x)

ax2.set_xticklabels(x, fontsize=20)

ax.set_xticklabels(dataset.columns[1:], rotation='vertical', fontsize=20)



ax.legend(fontsize=20)

plt.gcf().subplots_adjust(bottom=0.35)  

plt.show()
X_enc_mod = X_enc[:,[3,4,7,8,11,12,13,14,18,19,20,21]]

svc = SVC()

parameters = {'C':(200, 300,400), 'gamma':(0.01,0.05,0.1)}

clf_select_features = GridSearchCV(svc, parameters)

clf_select_features.fit(X_enc_mod, y_enc)

print(clf_select_features.best_score_)
X_enc_mod = X_enc[:, [3,6,7,8,10,18,21]]

svc = SVC()

parameters = {'C':(100,200,500), 'gamma':(0.1,1,100)}

clf_select_features = GridSearchCV(svc, parameters)

clf_select_features.fit(X_enc_mod, y_enc)

print(clf_select_features.best_score_)
X_enc_mod = X_enc[:, [3,6,7,8,10,18,21,4,19,20]]

svc = SVC()

parameters = {'C':(95,100,105), 'gamma':(0.5,0.6,0.7)}

clf_select_features = GridSearchCV(svc, parameters)

clf_select_features.fit(X_enc_mod, y_enc)

print(clf_select_features.best_score_)