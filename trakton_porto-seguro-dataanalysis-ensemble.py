# data mining

import numpy as np

import pandas as pd



# data visualization

import seaborn as sns

import missingno as msno

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



from subprocess import check_output 

print(check_output(["ls", "../input"]).decode("utf8"))
def gini(actual, pred):

    assert (len(actual) == len(pred))

    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)

    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

    totalLosses = all[:, 0].sum()

    giniSum = all[:, 0].cumsum().sum() / totalLosses



    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)





def gini_normalized(actual, pred):

    return gini(actual, pred) / gini(actual, actual)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score
df = pd.read_csv('../input/train.csv', na_values='-1')

test_df = pd.read_csv('../input/test.csv', na_values='-1')

df.head()
entries = df.shape[0]

plot = sns.countplot(x='target', data=df)

for p in plot.patches:

    plot.annotate('{:.2f}%'.format(100*p.get_height()/entries), (p.get_x()+ 0.3, p.get_height()+10000))
msno.matrix(df=df.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
print('Column \t\t Number of Null')

for column in df.columns:

    print('{}:\t {} ({:.2f}%)'.format(column,len(df[column][np.isnan(df[column])]), 100*len(df[column][np.isnan(df[column])])/entries))
corr = df.corr()

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
unwanted = df.columns[df.columns.str.startswith('ps_calc_')]

df = df.drop(unwanted, axis=1)

test_df = test_df.drop(unwanted, axis=1)

df.head()
cat_columns = [a for a in df.columns if a.endswith('cat')]



for col in cat_columns:

	dummy = pd.get_dummies(pd.Series(df[col]))

	df = pd.concat([df,dummy],axis=1)

	df = df.drop([col],axis=1)

    

for col in cat_columns:

	dummy = pd.get_dummies(pd.Series(test_df[col]))

	test_df = pd.concat([test_df,dummy],axis=1)

	test_df = test_df.drop([col],axis=1)

    

df.head()

class Ensemble(object):

    def __init__(self, kfold, stacker, models):

        self.kfold = kfold

        self.stacker = stacker

        self.models = models



    def fit_predict(self, x, y, test):

        x = np.array(x)

        y = np.array(y)

        t = np.array(test)

        

        train = np.zeros((x.shape[0], len(self.models)))

        test = np.zeros((t.shape[0], len(self.models)))

        

        skf = list(StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=2016).split(x, y))

        

        for i, model in enumerate(self.models):



            test_i = np.zeros((t.shape[0], self.kfold))



            for j, (train_idx, test_idx) in enumerate(skf):

                x_train = x[train_idx]

                y_train = y[train_idx]

                x_valid = x[test_idx]

                y_valid = y[test_idx]



                print ("Fit %s fold %d" % (str(model).split('(')[0], j+1))

                

                model.fit(x_train, y_train)

                y_train_pred = model.predict_proba(x_train)[:,1]

                y_pred = model.predict_proba(x_valid)[:,1]   

                

                print("[Train] Gini score: %.6lf" % gini_normalized(y_train, y_train_pred))

                print("[Test] Gini score: %.6lf\n" % gini_normalized(y_valid, y_pred))



                train[test_idx, i] = y_pred

                test_i[:, j] = model.predict_proba(t)[:,1]

            test[:, i] = test_i.mean(axis=1)



        self.stacker.fit(train, y)

        valid = self.stacker.predict_proba(train)[:,1]

        res = self.stacker.predict_proba(test)[:,1]

        print("Staker Gini Score: %.6lf" % gini_normalized(valid, y))

        return res
x = df.drop(['id', 'target'], axis=1)

y = df['target'].values

test_id = test_df['id']

test_df = test_df.drop('id', axis=1)
lgb_params = {

    'learning_rate': 0.02,

    'n_estimators': 1, # use 650 for real model

    'max_bin': 10,

    'subsample': 0.8,

    'subsample_freq': 10,

    'colsample_bytree': 0.8,

    'min_child_samples': 500,

    'random_state': 99

}



lgb_model = LGBMClassifier(**lgb_params)



lgb2_params = {

    'learning_rate': 0.02,

    'n_estimators': 1, #use 1090 for real model

    'colsample_bytree': 0.3,

    'subsample': 0.7,

    'subsample_freq': 2,

    'num_leaves': 16,

    'random_state': 99

}



lgb_model2 = LGBMClassifier(**lgb2_params)



lgb3_params = {

    'n_estimators': 1, #use 1100 for real model

    'max_depth': 4,

    'learning_rate': 0.02,

    'random_state': 99

}



lgb_model3 = LGBMClassifier(**lgb3_params)



log_model = LogisticRegression()
stack = Ensemble(kfold=3,

        stacker = log_model,

        models = (lgb_model, lgb_model2, lgb_model3))        

        

y_pred = stack.fit_predict(x, y, test_df)
sub = pd.DataFrame()

sub['id'] = test_id

sub['target'] = y_pred

sub.to_csv('output.csv', index=False)