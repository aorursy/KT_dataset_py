import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline

print(os.listdir("../input"))



inputData = pd.read_csv(r"../input/eeg_clean.csv");
print(inputData.dtypes)

print(inputData.columns)

print("Data shape:",inputData.shape)

print(inputData.head())

print(inputData.describe())

print(inputData.info())

# Check for any nulls

print(inputData.isnull().sum())

# Lets convert the open/closed category for eye into integers
inputData['eye']=inputData["eye"].astype('category')

inputData["eye"] = inputData["eye"].cat.codes

print ("************************************")

print ("EXPERIMENT WITH TEST AND TRAIN SPLIT")

print ("************************************")
from sklearn.model_selection import train_test_split

splitRatio = 0.2

train , test = train_test_split(inputData,test_size = splitRatio,random_state = 123,shuffle=True)



plt.figure(figsize=(12,6))

plt.subplot(121)

train["eye"].value_counts().plot.pie(labels = ["1-open","0-closed"],

                                              autopct = "%1.0f%%",

                                              shadow = True,explode=[0,.1])

plt.title("proportion of target class in train data")

plt.ylabel("")

plt.subplot(122)

test["eye"].value_counts().plot.pie(labels = ["1-open","0-closed"],

                                             autopct = "%1.0f%%",

                                             shadow = True,explode=[0,.1])

plt.title("proportion of target class in test data")

plt.ylabel("")

plt.show()



#Seperating Predictor and target variables

train_X = train[[x for x in train.columns if x not in ["eye"]]]

train_Y = train[["eye"]]

test_X  = test[[x for x in test.columns if x not in ["eye"]]]

test_Y  = test[["eye"]]

import lightgbm as lgbm

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc
# create dataset for lightgbm


lgb_train = lgbm.Dataset(train_X, train_Y)

lgb_eval = lgbm.Dataset(test_X, test_Y, reference=lgb_train)
params = {

    'objective' :'binary',

    'tree_learner':'data',

    'learning_rate' : 0.1,

    'num_leaves' :99 ,

    'feature_fraction': 0.8, 

    'bagging_fraction': 0.8, 

    'bagging_freq':1,

    'boosting_type' : 'gbdt',

    'metric': 'binary_logloss'

}

classifier = lgbm.train(params, lgb_train, 700)

predictions = classifier.predict(test_X)

predictedLabels = (predictions>0.35).astype(int)

print ("\naccuracy_score :",accuracy_score(test_Y,predictedLabels))

print ("\nclassification report :\n",(classification_report(test_Y,predictedLabels)))
plt.figure(figsize=(13,10))

plt.subplot(221)

sns.heatmap(confusion_matrix(test_Y,predictedLabels),annot=True,fmt = "d",linecolor="k",linewidths=3)

plt.title("CONFUSION MATRIX",fontsize=20)

predicting_probabilites = predictions

fpr,tpr,thresholds = roc_curve(test_Y,predicting_probabilites)

plt.subplot(222)

plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")

plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")

plt.legend(loc = "best")

plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)    

dataframe = pd.DataFrame(classifier.feature_importance(),train_X.columns).reset_index()

dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})

dataframe = dataframe.sort_values(by="coefficients",ascending = False)

plt.subplot(223)

ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")

plt.title("FEATURE IMPORTANCES",fontsize =20)

for i,j in enumerate(dataframe["coefficients"]):

    ax.text(.011,i,j,weight = "bold")

plt.show()




