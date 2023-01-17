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



import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import pointbiserialr, spearmanr

from sklearn.cross_validation import cross_val_score

from sklearn.feature_selection import SelectKBest



from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, roc_auc_score



# for logit regression. 

# statsmodel is chosen because it outputs descriptive stats for the model

import statsmodels.api as sm



# for SVM

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



# for random forest

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv("../input/adult.csv")

data.head()
col_names = data.columns

num_data = data.shape[0]

for c in col_names:

    num_non = data[c].isin(["?"]).sum()

    if num_non > 0:

        print (c)

        print (num_non)

        print ("{0:.2f}%".format(float(num_non) / num_data * 100))

        print ("\n")
data = data[data["workclass"] != "?"]

data = data[data["occupation"] != "?"]

data = data[data["native-country"] != "?"]



data.shape
# descriptive stats for numerical fields

data.describe()
# frequency for categorical fields 

category_col =['workclass', 'race', 'education','marital-status', 'occupation',

               'relationship', 'gender', 'native-country', 'income'] 

for c in category_col:

    print (c)

    print (data[c].value_counts())
data["income"].value_counts()[0] / data.shape[0]
data["income"].value_counts()[1] / data.shape[0]
data.replace(['Divorced', 'Married-AF-spouse', 

              'Married-civ-spouse', 'Married-spouse-absent', 

              'Never-married','Separated','Widowed'],

             ['not married','married','married','married',

              'not married','not married','not married'], inplace = True)
for col in category_col:

    b, c = np.unique(data[col], return_inverse=True) 

    data[col] = c



data.head()
col_names = data.columns



param=[]

correlation=[]

abs_corr=[]



for c in col_names:

    #Check if binary or continuous

    if c != "income":

        if len(data[c].unique()) <= 2:

            corr = spearmanr(data['income'],data[c])[0]

        else:

            corr = pointbiserialr(data['income'],data[c])[0]

        param.append(c)

        correlation.append(corr)

        abs_corr.append(abs(corr))



#Create dataframe for visualization

param_df=pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr})



#Sort by absolute correlation

param_df=param_df.sort_values(by=['abs_corr'], ascending=False)



#Set parameter name as index

param_df=param_df.set_index('parameter')



param_df
scoresCV = []

scores = []



for i in range(1,len(param_df)):

    new_df=data[param_df.index[0:i+1].values]

    X = new_df.ix[:,1::]

    y = new_df.ix[:,0]

    clf = DecisionTreeClassifier()

    scoreCV = cross_val_score(clf, X, y, cv= 10)

    scores.append(np.mean(scoreCV))

    

plt.figure(figsize=(15,5))

plt.plot(range(1,len(scores)+1),scores, '.-')

plt.axis("tight")

plt.title('Feature Selection', fontsize=14)

plt.xlabel('# Features', fontsize=12)

plt.ylabel('Score', fontsize=12)

plt.grid();
best_features=param_df.index[0:4].values

print('Best features:\t',best_features)
predictors = ['age','workclass','education','educational-num',

              'marital-status', 'occupation','relationship','race','gender',

              'capital-gain','capital-loss','hours-per-week', 'native-country']



#predictors = ['marital-status', 'educational-num', 'relationship', 'age']



high_income = data[data['income'] == 1]

low_income = data[data['income'] == 0]



# stratified sampling

#80% to train set

train = pd.concat([high_income.sample(frac=0.8, random_state=1),

                   low_income.sample(frac=0.8, random_state=1)]) 

y_train = train["income"]

X_train = train[predictors]



#10% to test set

test = pd.concat([high_income.sample(frac=0.1, random_state=1), 

                  low_income.sample(frac=0.1, random_state=1)])

y_test = test["income"]

X_test = test[predictors]



#10% to CV set

cross = pd.concat([high_income.sample(frac=0.1, random_state=2), 

                   low_income.sample(frac=0.1, random_state=2)])

y_cross = cross["income"]

X_cross = cross[predictors]
#train

print("train set result\n")

logit_train = sm.Logit(y_train, X_train) 

result_train = logit_train.fit()



y_train_pred = result_train.predict(X_train) 

y_train_pred = (y_train_pred > 0.5).astype(int) 

acc = accuracy_score(y_train, y_train_pred) 

print("ACC=%f" % (acc))

auc = roc_auc_score(y_train, y_train_pred) 

print("AUC=%f" % (auc))



print("\n CV set result\n")

y_cross_pred = result_train.predict(X_cross) 

y_cross_pred = (y_cross_pred > 0.5).astype(int)

acc = accuracy_score(y_cross, y_cross_pred) 

print("ACC=%f" % (acc))

auc = roc_auc_score(y_cross, y_cross_pred) 

print ("AUC=%f" % (auc))



print("\n test set result\n")

y_test_pred = result_train.predict(X_test) 

y_test_pred = (y_test_pred > 0.5).astype(int) 

acc = accuracy_score(y_test, y_test_pred)

print("ACC=%f" % (acc))

auc = roc_auc_score(y_test, y_test_pred)

print("AUC=%f" % (auc))
predictors = ['age','workclass','education','educational-num',

              'marital-status', 'occupation','relationship','race','gender',

              'capital-gain','capital-loss','hours-per-week', 'native-country']



predictors = ['marital-status', 'educational-num', 'relationship', 'age']



pred_data = data[predictors] #X

target = data["income"] #y
algorithms = [ 

    #linear kernel

    [Pipeline([('scaler',StandardScaler()), 

               ('svc',LinearSVC(random_state=1))]), predictors],

    #rbf kernel

    [Pipeline([('scaler',StandardScaler()),

               ('svc',SVC(kernel="rbf", random_state=1))]), predictors],

    #polynomial kernel

    [Pipeline([('scaler',StandardScaler()),

               ('svc',SVC(kernel='poly', random_state=1))]), predictors],

    #sigmoidf kernel

    [Pipeline([('scaler',StandardScaler()),

               ('svc',SVC(kernel='sigmoid', random_state=1))]), predictors]

]
alg_acc = {}

alg_auc = {}

for alg, predictors in algorithms:

    alg_acc[alg] = 0

    alg_auc[alg] = 0

i=0



pred_data = data[predictors] #X

target = data["income"] #y



#stratified sampling

#random_state=1: we get the same splits every time we run this

# sss = StratifiedShuffleSplit(target, 10, test_size=0.1, random_state=1) 

sss = StratifiedShuffleSplit(target, 1, test_size=0.1, random_state=1) 

for train_index, test_index in sss:

    i += 1

    train_data = data.iloc[train_index]

    test_data = data.iloc[test_index]

    train_data = pd.concat([train_data,

                            train_data[train_data["income"]==1],

                            train_data[train_data["income"]==1]]) 

    X_train, X_test = train_data[predictors], test_data[predictors] 

    y_train, y_test = train_data["income"], test_data["income"]

    

    #Make predictions for each algorithm on each fold for alg, predictors in algorithms:

    for alg, predictors in algorithms:

        alg.fit(X_train, y_train)

        y_pred = alg.predict(X_test)

        acc_score = accuracy_score(y_test, y_pred) 

        print(acc_score)

        alg_acc[alg] += acc_score

        auc_score = roc_auc_score(y_test, y_pred) 

        print(auc_score)

        alg_auc[alg] += auc_score



for alg, predictors in algorithms:

    alg_acc[alg] /= 1

    alg_auc[alg] /= 1

    print ("## %s ACC=%f" % (alg, alg_acc[alg]))

    print ("## %s AUC=%f" % (alg, alg_auc[alg]))
#Bagging

tree_count = 10 

bag_proportion = 0.6 

predictions = []



sss = StratifiedShuffleSplit(target, 1, test_size=0.1, random_state=1) 

for train_index, test_index in sss:

    train_data = data.iloc[train_index] 

    test_data = data.iloc[test_index]

    

    for i in range(tree_count):

        bag = train_data.sample(frac=bag_proportion, replace = True, random_state=i)

        X_train, X_test = bag[predictors], test_data[predictors]

        y_train, y_test = bag["income"], test_data["income"]

        clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=75) 

        clf.fit(X_train, y_train) 

        predictions.append(clf.predict_proba(X_test)[:,1])



combined = np.sum(predictions, axis=0)/10 

rounded= np.round(combined)



print(accuracy_score(rounded, y_test)) 

print(roc_auc_score(rounded, y_test))