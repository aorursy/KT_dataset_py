import pandas as pd

import seaborn as sns

import numpy as np

from scipy.stats import norm
data = pd.read_csv("../input/hmeq.csv")
data.shape
data.head()
data.isna().sum()
df = data.dropna()
df.shape
df_con_bad  = df.drop("REASON",axis = 1 )

df_con_bad = df_con_bad.drop("JOB",axis = 1)
from pylab import rcParams

rcParams['figure.figsize'] = 10, 10

df_con_bad_corr = df_con_bad.drop("BAD",axis = 1)  

corr = df_con_bad_corr.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
dic = {"LOAN":df["LOAN"],"BAD":df["BAD"],"MORTDUE":df["MORTDUE"],"VALUE":df["VALUE"],"YOJ":df["YOJ"]}

rcParams['figure.figsize'] = 5, 5

df_pair = pd.DataFrame(dic)

sns.pairplot(df_pair,vars=['LOAN', 'MORTDUE',"VALUE","YOJ"],hue="BAD")
df[df["BAD"]==0].shape
df[df["BAD"]==1].shape
df_set_bad = df.set_index("BAD")
import numpy as np

import matplotlib.pyplot as plt

rcParams['figure.figsize'] = 8, 5

# data to plot

n_groups = 3

df_set_bad_mean_undefault = df_set_bad.loc[0].mean()

df_set_bad_mean_default = df_set_bad.loc[1].mean()



# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.35

opacity = 0.8

 

rects1 = plt.bar(index, df_set_bad_mean_undefault[0:3], bar_width,alpha=opacity,color='b',label='Undefault')

 

rects2 = plt.bar(index + bar_width, df_set_bad_mean_default[0:3], bar_width,alpha=opacity,color='g',label='Default')

 

plt.xlabel('Mean of each variables')

plt.ylabel('$ value')

plt.title('Comparison of mean of being undefault and default based on LOAN,MORTDUE and VALUE')

plt.xticks(index + bar_width, ("LOAN","MORTDUE","VALUE","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC"))

plt.legend()

 

plt.tight_layout()

plt.show()
df_set_bad.loc[0].mean()
df_set_bad.loc[1].mean()
#pad=0.3, w_pad=4, h_pad=1.0

import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 30, 15

fig, axs = plt.subplots(3,3)

plt.tight_layout()

fig.subplots_adjust(top=0.88)

sns.boxplot(x="BAD", y="LOAN", data=df,ax=axs[0,0])

axs[0,0].set_title(" Amount of the loan requesed for a group of being default and undefault\n'",fontsize=14)

sns.boxplot(x="BAD", y="MORTDUE", data=df,ax=axs[0,1])

axs[0,1].set_title(" Amount due on existing mortgage for a group of being default and undefault\n'",fontsize=14)

sns.boxplot(x="BAD", y="VALUE", data=df,ax=axs[0,2])

axs[0,2].set_title(" Value of current property for a group of being default and undefault\n'",fontsize=14)



sns.boxplot(x="BAD", y="YOJ", data=df,ax=axs[1,0])

axs[1,0].set_title("Years at present job for a group of being default and undefault\n'",fontsize=14)



sns.boxplot(x="BAD", y="DEROG", data=df,ax=axs[1,1])

axs[1,1].set_title(" Number of major derogatory report for a group of being default and undefault\n'",fontsize=14)



sns.boxplot(x="BAD", y="DELINQ", data=df,ax=axs[1,2])

axs[1,2].set_title("     Number of delinquent credit lines  for a group of being default and undefault\n'",fontsize=14)



sns.boxplot(x="BAD", y="CLAGE", data=df,ax=axs[2,0])

axs[2,0].set_title(" Age of oldest credit line in months for a group of being default and undefault\n'",fontsize=14)





sns.boxplot(x="BAD", y="CLNO", data=df,ax=axs[2,1])

axs[2,1].set_title(" Number of credit lines for a group of being default and undefault\n'",fontsize=14)





sns.boxplot(x="BAD", y="DEBTINC", data=df,ax=axs[2,2])

axs[2,2].set_title("Debt-to-income ratio for a group of being default and undefault\n'",fontsize=14)



plt.tight_layout()

plt.show()
df_set_job = df.set_index("JOB")
df_set_job["BAD"].value_counts()
import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 20, 10



fig, axs = plt.subplots(2,4)

plt.tight_layout(pad=0.5, w_pad=4, h_pad=1.0)



sns.boxplot(x="JOB", y="LOAN", data=df,ax=axs[0,0])

sns.boxplot(x="JOB", y="MORTDUE", data=df,ax=axs[0,1])

sns.boxplot(x="JOB", y="VALUE", data=df,ax=axs[0,2])

sns.boxplot(x="JOB", y="YOJ", data=df,ax=axs[0,3])

sns.boxplot(x="JOB", y="CLAGE", data=df,ax=axs[1,0])

sns.boxplot(x="JOB", y="NINQ", data=df,ax=axs[1,1])

sns.boxplot(x="JOB", y="CLNO", data=df,ax=axs[1,2])
ct = pd.crosstab(df.BAD,df.JOB,margins=True) #making cross table  

ct
from scipy.stats import chi2_contingency

chi2, p, dof, ex = chi2_contingency(ct)

print("chi2 = ", chi2)

print("p-val = ", p)

print("degree of freedom = ",dof)

print("Expected:")

pd.DataFrame(ex)

import numpy as np

import pandas as pd

import scipy as sp

import sklearn as sk

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

import sklearn.ensemble as skens

import sklearn.metrics as skmetric

import sklearn.naive_bayes as sknb

import sklearn.tree as sktree

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style='white', color_codes=True, font_scale=1.3)

import sklearn.externals.six as sksix

import IPython.display as ipd

from sklearn.model_selection import cross_val_score

from sklearn import metrics

import os
import warnings

warnings.filterwarnings('ignore')
df_train,df_test = train_test_split(df, test_size=0.3)

df_train_drop_cate = df_train.drop("REASON",axis=1) #droping the categorical variables 

df_test_drop_cate = df_test.drop("REASON",axis=1)

df_train_drop_cate = df_train_drop_cate.drop("JOB",axis=1)

df_test_drop_cate = df_test_drop_cate.drop("JOB",axis=1)

df_train_drop_cate = df_train_drop_cate.drop("BAD",axis=1)

df_test_drop_cate = df_test_drop_cate.drop("BAD",axis=1)
df_train.shape
s = df_train_drop_cate
df_train_drop_cate_show = s

df_train_drop_cate_show["BAD"] = df_train["BAD"]
df_train_drop_cate = df_train_drop_cate.drop("BAD",axis=1)
dt_model = sktree.DecisionTreeClassifier(max_depth=3,

                                         criterion='entropy')

dt_model.fit(df_train_drop_cate,df_train_drop_cate_show.BAD)
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import graphviz



export_graphviz(dt_model, out_file="tree_dt_model.dot",  

                filled=True, rounded=True,

                special_characters=True,feature_names=df_train_drop_cate.columns)

#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

#Image(filename = 'tree_dt_model.dot')
!dot -Tpng tree_dt_model.dot -o tree_dt_model.png -Gdpi=600
from IPython.display import Image

Image(filename = 'tree_dt_model.png')
predicted_labels = dt_model.predict(df_train_drop_cate)

df_train_drop_cate_show['predicted_dt_tree'] = predicted_labels
len(df_train_drop_cate_show[df_train_drop_cate_show['BAD'] == df_train_drop_cate_show["predicted_dt_tree"]])/len(df_train_drop_cate_show)
def comparePlot(input_frame,real_column,predicted_column):

    df_a = input_frame.copy()

    df_b = input_frame.copy()

    df_a['label_source'] = 'BAD'

    df_b['label_source'] = 'Classifier'

    df_a['label'] = df_a[real_column]

    df_b['label'] = df_b[predicted_column].apply(lambda x: 'Predict %s'%x)

    df_c = pd.concat((df_a, df_b), axis=0, ignore_index=True)

    sns.lmplot(x='DEBTINC', y='CLAGE', col='label_source',

               hue='label', data=df_c, fit_reg=False, size=3);
comparePlot(df_train_drop_cate_show,"BAD","predicted_dt_tree")
rf_model = skens.RandomForestClassifier(n_estimators=10,oob_score=True, criterion='entropy')

rf_model.fit(df_train_drop_cate,df_train.BAD)
feat_importance = rf_model.feature_importances_

pd.DataFrame({'Feature Importance':feat_importance},

            index=df_train_drop_cate.columns).plot(kind='barh')

rcParams['figure.figsize'] = 10, 5
predicted_labels = rf_model.predict(df_train_drop_cate)

df_train_drop_cate_show['predicted_rf_tree'] = predicted_labels

comparePlot(df_train_drop_cate_show,"BAD","predicted_rf_tree")
param_grid = {

                 'n_estimators': [5, 10, 15, 20, 25],

                 'max_depth': [2, 5, 7, 9],

             }
from sklearn.model_selection import GridSearchCV
grid_clf = GridSearchCV(rf_model, param_grid, cv=10)

grid_clf.fit(df_train_drop_cate,df_train.BAD)
grid_clf.best_estimator_
grid_clf.best_params_ 
#Turing the model into the one with max_depth': 9, 'n_estimators': 20

rf_model2 = skens.RandomForestClassifier(n_estimators=20,oob_score=True,max_depth=9, criterion='entropy')

rf_model2.fit(df_train_drop_cate,df_train.BAD)

predicted_labels2 = rf_model.predict(df_train_drop_cate)

df_train_drop_cate_show['predicted_rf_tree2'] = predicted_labels2
len(df_train_drop_cate_show[df_train_drop_cate_show['BAD'] == df_train_drop_cate_show["predicted_rf_tree2"]])/len(df_train_drop_cate_show)
df["is_default"] = np.where(df["BAD"] == 1,"default","not_default")

df_bay_train,df_bay_test = train_test_split(df, test_size=0.3)


gnb_model = sknb.GaussianNB()



gnb_model.fit(df_bay_train[['DEBTINC']],df_bay_train['is_default'])
# test the model

y_pred = gnb_model.predict(df_bay_test[['DEBTINC']])

df_bay_test['predicted_nb'] = y_pred
comparePlot(df_bay_test,"is_default","predicted_nb")
len(df_bay_test[df_bay_test['is_default'] == df_bay_test["predicted_nb"]])/len(df_bay_test)