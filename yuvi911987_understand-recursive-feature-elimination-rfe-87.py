from xgboost import XGBClassifier
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from numpy import *

#from IPython.core.display import display, HTML



import warnings

warnings.filterwarnings('ignore')

warnings.simplefilter('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



sns.set(style='white', context='notebook', palette='deep')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_kaggle = pd.read_csv("../input/adult-census-income/adult.csv")

data_kaggle.head()
#Total rows, column

data_kaggle.shape
#Displaying datatypes and number of non-missing value for each column 

data_kaggle.info()
# Dropping the duplicate Rows



print("shape with duplicate rows: ",data_kaggle.shape)

data_kaggle = data_kaggle.drop_duplicates(keep = 'first')

print("shape without duplicate rows: ",data_kaggle.shape)
data_kaggle.describe()
#Displaying names of all columns

print(data_kaggle.columns)

#Displaying levels in all categorical columns

print("workclass :",data_kaggle.workclass.unique())

print("education :",data_kaggle.education.unique())

print("marital status :",data_kaggle['marital.status'].unique())

print("occupation :",data_kaggle.occupation.unique())

print("relationship :",data_kaggle.relationship.unique())

print("race :",data_kaggle.race.unique())

print("sex :",data_kaggle.sex.unique())

print("native country :",data_kaggle['native.country'].unique())

#Displaying target field and distribution for two classes

print(data_kaggle.income.unique())

print(data_kaggle.income.value_counts())
# Count of >50K & <=50K

sns.countplot(data_kaggle['income'],label="Count")

#sns.plt.show()
# Histogram Distribution of all numeric fields of the Dataset

distribution = data_kaggle.hist(edgecolor = 'black', linewidth = 1.2, color = 'b')

fig = plt.gcf()

fig.set_size_inches(12,12)

plt.show()
#Plotting levels in all categorical columns

data_kaggle.sex.value_counts().head(10).plot.bar()
data_kaggle.workclass.value_counts().plot.bar()
data_kaggle.education.value_counts().plot.bar()
data_kaggle['education.num'].value_counts().plot.bar()
data_kaggle['marital.status'].value_counts().plot.bar()
data_kaggle.occupation.value_counts().plot.bar()
data_kaggle.relationship.value_counts().plot.bar()
data_kaggle.race.value_counts().plot.bar()
data_kaggle.sex.value_counts().plot.bar()
data_kaggle['native.country'].value_counts().plot.bar()
#Plotting Sex along with the target field income. This has shown that most of the females are earning lesser than 50000 dollars

#Thus Sex can be a good predictor of target class.

pd.crosstab(data_kaggle['sex'],data_kaggle['income']).plot(kind="bar",figsize=(15,6) )

plt.title('Income  for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.ylabel('Frequency')

plt.show()

#Getting count of '?' in occupation

count_occu = data_kaggle[data_kaggle.occupation == '?'].occupation.count()

print(count_occu)

#Getting count of '?' in workclass

count_work = data_kaggle[data_kaggle.workclass == '?'].workclass.count()

print(count_work)
#Getting count of '?' in occupation

data_kaggle[data_kaggle.occupation == '?'].workclass.value_counts()

#Getting count of '?' in occupation

data_kaggle[data_kaggle.workclass == '?'].occupation.value_counts()

# This will replace "?" with "unemployed" 

data_kaggle.replace(to_replace ="?", 

                 value ="unemployed", inplace=True)
data_kaggle[data_kaggle['native.country'] == 'unemployed'].shape
# This will replace "?" with "unemployed" 

data_kaggle['native.country'].replace(to_replace ="unemployed", 

                 value ="unknown", inplace=True)
print(data_kaggle[data_kaggle['capital.gain']==0].shape)

print(data_kaggle[data_kaggle['capital.loss']==0].shape)
data_kaggle[data_kaggle['capital.gain']==0].filter(['capital.gain','capital.loss'])
data_kaggle[data_kaggle['capital.gain']==0]['capital.loss'].value_counts()
data_kaggle[data_kaggle['capital.loss']==0]['capital.gain'].value_counts()
#Majority values are 0 for both gain and loss

data_kaggle[(data_kaggle['capital.gain']==0) & (data_kaggle['capital.loss'] == 0)].shape
#Merging two columns - such that a negative value means capital loss and positive value means capital gain

data_kaggle['capital.flow'] =  data_kaggle['capital.gain'] - data_kaggle['capital.loss']
data_kaggle['capital.flow'].value_counts()
# Histogram Distribution of newly created capital.flow of the Dataset

distribution = data_kaggle['capital.flow'].hist(edgecolor = 'black', linewidth = 1.2, color = 'b')

fig = plt.gcf()

fig.set_size_inches(12,12)

plt.show()
 #This heatmap shows the Correlation between the different variables

plt.rcParams['figure.figsize'] = [10,7]

sns.heatmap(data_kaggle.filter(['age','fnlwgt','capital.flow','hours.per.week']).corr(), annot = True);
data_kaggle.columns

data_kaggle.values
# Feature Selection with Univariate Statistical Tests - for Numeric inputs

from pandas import read_csv

from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

# load data

names = ['age','fnlwgt','capital.flow','hours.per.week']

X= data_kaggle[names]

Y= data_kaggle['income']

# Performing feature extraction using ANOVA method

test = SelectKBest(score_func=f_classif, k=2)

fit = test.fit(X, Y)

# summarize scores

feature_list_num = fit.scores_

print(fit.scores_)
#Label Encoder - Encoding or feature engineering for categorical (non-ordinal) variables

from sklearn.preprocessing import LabelEncoder

#Creating a copy of my dataframe

data_kaggle_cat = data_kaggle

categorical_features = ['workclass','marital.status','occupation','relationship','race','sex','native.country']



label_encoder_feat = {}

for i, feature in enumerate(categorical_features):

    label_encoder_feat[feature] = LabelEncoder()

    data_kaggle_cat[feature] = label_encoder_feat[feature].fit_transform(data_kaggle_cat[feature])



data_kaggle_cat.head()
# Feature Selection with Univariate Statistical Tests - for Categorical inputs

from pandas import read_csv

from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import chi2



# load data

names = ['workclass','education.num','marital.status','occupation','relationship','race','sex','native.country']

#names = ['education.num','marital.status','relationship','race','sex','native.country']



X= data_kaggle_cat[names]

Y= data_kaggle_cat['income']

# Performing feature extraction using Chi-square method

test = SelectKBest(score_func=chi2, k=2)

fit = test.fit(X, Y)

# summarize scores

feature_list_cat = fit.scores_

print(fit.scores_)
data_kaggle.columns
data_kaggle.head(5)
#Encoding Target variable as 0 or 1 

#Changing Positive to 1 and Negative to 0 for ease of processing

data_kaggle.loc[data_kaggle["income"] == "<=50K", "income"] = 1

data_kaggle.loc[data_kaggle["income"] == ">50K", "income"] = 0

data_kaggle.head(5)

data_kaggle.income.value_counts()

# 1 signifies income is less than equal to 50,000 $

# 0 signifies income is more than 50,000 $
#Creating the final dataset after all feature engineering

feature_list = ['age','capital.flow','hours.per.week', 'relationship', 'education.num', 'marital.status','sex','income']

data_kaggle_feature1 = data_kaggle[feature_list]

data_kaggle_feature1.head(5)
data_kaggle_feature1.income =data_kaggle_feature1.income.astype('int')
#Label Encoder - Encoding or feature engineering for categorical (non-ordinal) variables

from sklearn.preprocessing import LabelEncoder

#Creating a copy of my dataframe

#data_kaggle_cat = data_kaggle

categorical_features = ['marital.status','relationship','sex']



label_encoder_feat = {}

for i, feature in enumerate(categorical_features):

    label_encoder_feat[feature] = LabelEncoder()

    data_kaggle_feature1[feature] = label_encoder_feat[feature].fit_transform(data_kaggle_feature1[feature])



data_kaggle_feature1.head()
#Creating Training(25,000)+Validation(15,000) and Test(10,000) sttartefied dataset - a split of 50-30-20% respectively

#First splitting dataset of 50,000 instances into training (80%) and test (20%)

from sklearn.model_selection import train_test_split

X_training, X_test, Y_training, Y_test = train_test_split(data_kaggle_feature1.iloc[:,0:-1], data_kaggle_feature1.income,

                                                    stratify=data_kaggle_feature1.income, 

                                                    test_size=0.10)







print("Shape of train split :",X_training.shape,Y_training.shape)

print("Shape of test split :",X_test.shape,Y_test.shape)
# scikit-learn k-fold cross-validation

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from itertools import compress

from numpy import array

from sklearn.model_selection import KFold, StratifiedKFold

from statistics import mode

from operator import itemgetter 

from sklearn.ensemble import GradientBoostingClassifier







# Prepare for 10-fold cross validation

skf = StratifiedKFold(n_splits=10)

skf.get_n_splits(X_training, Y_training)

accu_kfold_list = []

counter_kfold =0

# enumerate splits

for train, val in skf.split(X_training, Y_training):

    counter_kfold+=1

    #print("KFold validation iteration :",counter_kfold)

    X_train, X_val = X_training.iloc[train], X_training.iloc[val]

    Y_train, Y_val = Y_training.iloc[train], Y_training.iloc[val]





    # create a base classifier used to evaluate a subset of attributes

    accu_list = []

    for num_of_vars in range(1,len(data_kaggle_feature1.columns)):

        

        #Logistic Regression

        logreg = LogisticRegression()

        # create the RFE model and select 3 attributes

        rfe = RFE(logreg, num_of_vars)

        rfe = rfe.fit(X_train, Y_train)



        # summarize the selection of the attributes

        #print("Number of variables used :",num_of_vars)

        col_list = data_kaggle_feature1.columns.to_list()[:-1]

        #print(col_list)

        col_bool = rfe.support_

        #print(col_bool)

        col_list = list(compress(col_list, col_bool))  

        #print("Variables used in building this classifier :",col_list)



        # fit

        logreg.fit(X_train.filter(col_list), Y_train)



        # predict

        Y_pred = logreg.predict(X_val.filter(col_list))



        accuracy = accuracy_score(Y_val, Y_pred)

        #print('LogReg %s' % accuracy)



        #Adding the accuracy to list

        accu_list.append(accuracy)



    #print(accu_list)

    accu_kfold_list.append(accu_list)





#List of all accuracies

#print(accu_kfold_list)



#Creating empty lists to calculate which variable configuration produced maximum acccuracy

lst_rfe = []

lst_acu = []

for lst in accu_kfold_list:

    print("Index or number of variables used to build a model:",lst.index(max(lst))+1, "Max accuracy value :",max(lst) )

    lst_rfe.append(lst.index(max(lst))+1)

    lst_acu.append(max(lst))

    

print("List of 10 accuracies :",lst_acu)

print("List of indices or number of variables used :" ,lst_rfe)  

mode_val = max(set(lst_rfe), key=lst_rfe.count)

print("The Final number of variables that i am using to build and report best model :",mode_val)

indices = [i for i, x in enumerate(lst_rfe) if x == mode_val]

print("The indices of most occuring k value :",indices)



# using operator.itemgetter() to get elements from list  

res_list = list(itemgetter(*indices)(lst_acu))

print("The accuracy values present at these indices :",res_list)

print("The Final mean cross validation accuracy for reporting purpose is :",mean(res_list))

#print("The number of variables used in the best model :",mode(lst_avg))
# scikit-learn k-fold cross-validation

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from itertools import compress

from numpy import array

from sklearn.model_selection import KFold, StratifiedKFold

from statistics import mode

from operator import itemgetter 

from sklearn.ensemble import GradientBoostingClassifier





# Prepare for 10-fold cross validation

skf = StratifiedKFold(n_splits=10)

skf.get_n_splits(X_training, Y_training)

accu_kfold_list = []

counter_kfold =0

# enumerate splits

for train, val in skf.split(X_training, Y_training):

    counter_kfold+=1

    #print("KFold validation iteration :",counter_kfold)

    X_train, X_val = X_training.iloc[train], X_training.iloc[val]

    Y_train, Y_val = Y_training.iloc[train], Y_training.iloc[val]





    # create a base classifier used to evaluate a subset of attributes

    accu_list = []

    for num_of_vars in range(1,len(data_kaggle_feature1.columns)):

        

        # Gradient Boosting Algorithm - fit

        gbc = GradientBoostingClassifier()



        # create the RFE model and select 3 attributes

        rfe = RFE(gbc, num_of_vars)

        rfe = rfe.fit(X_train, Y_train)



        

        # summarize the selection of the attributes

        #print("Number of variables used :",num_of_vars)

        col_list = data_kaggle_feature1.columns.to_list()[:-1]

        #print(col_list)

        col_bool = rfe.support_

        #print(col_bool)

        col_list = list(compress(col_list, col_bool))  

        #print("Variables used in building this classifier :",col_list)



        # fit

        #logreg.fit(X_train.filter(col_list), Y_train)

        gbc.fit(X_train.filter(col_list), Y_train)

        accuracy = gbc.score(X_val.filter(col_list), Y_val)

        #print('GBC %s' % accuracy)



        #Adding the accuracy to list

        accu_list.append(accuracy)



    #print(accu_list)

    accu_kfold_list.append(accu_list)





#List of all accuracies

#print(accu_kfold_list)



#Creating empty lists to calculate which variable configuration produced maximum acccuracy

lst_rfe = []

lst_acu = []

for lst in accu_kfold_list:

    print("Index or number of variables used to build a model:",lst.index(max(lst))+1, "Max accuracy value :",max(lst) )

    lst_rfe.append(lst.index(max(lst))+1)

    lst_acu.append(max(lst))

    

print("List of 10 accuracies :",lst_acu)

print("List of indices or number of variables used :" ,lst_rfe)

mode_val = max(set(lst_rfe), key=lst_rfe.count)

print("The Final number of variables that i am using to build and report best model :",mode_val)

indices = [i for i, x in enumerate(lst_rfe) if x == mode_val]

print("The indices of most occuring k value :",indices)



# using operator.itemgetter() to get elements from list  

res_list = list(itemgetter(*indices)(lst_acu))

print("The accuracy values present at these indices :",res_list)

print("The Final mean cross validation accuracy for reporting purpose is :",mean(res_list))

#print("The number of variables used in the best model :",mode(lst_avg))
# scikit-learn k-fold cross-validation

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from itertools import compress

from numpy import array

from sklearn.model_selection import KFold, StratifiedKFold

from statistics import mode

from operator import itemgetter 

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier





# Prepare for 10-fold cross validation

skf = StratifiedKFold(n_splits=10)

skf.get_n_splits(X_training, Y_training)

accu_kfold_list = []

counter_kfold =0

# enumerate splits

for train, val in skf.split(X_training, Y_training):

    counter_kfold+=1

    #print("KFold validation iteration :",counter_kfold)

    X_train, X_val = X_training.iloc[train], X_training.iloc[val]

    Y_train, Y_val = Y_training.iloc[train], Y_training.iloc[val]





    # create a base classifier used to evaluate a subset of attributes

    accu_list = []

    for num_of_vars in range(1,len(data_kaggle_feature1.columns)):

        

        # Random Forest Algorithm 

        R_forest = RandomForestClassifier(n_estimators = 200)



        # create the RFE model 

        rfe = RFE(R_forest, num_of_vars)

        rfe = rfe.fit(X_train, Y_train)



        

        # summarize the selection of the attributes

        #print("Number of variables used :",num_of_vars)

        col_list = data_kaggle_feature1.columns.to_list()[:-1]

        #print(col_list)

        col_bool = rfe.support_

        #print(col_bool)

        col_list = list(compress(col_list, col_bool))  

        #print("Variables used in building this classifier :",col_list)



        # Training the model - Fitting

        model_random = R_forest.fit(X_train.filter(col_list), Y_train)

        

        # Predictions

        pred_random = model_random.predict(X_val.filter(col_list))

        accuracy = accuracy_score(Y_val, pred_random)

        #print ("The accuracy of Random Forest model is : ",accuracy)



        #Adding the accuracy to list

        accu_list.append(accuracy)



    #print(accu_list)

    accu_kfold_list.append(accu_list)





#List of all accuracies

#print(accu_kfold_list)



#Creating empty lists to calculate which variable configuration produced maximum acccuracy

lst_rfe = []

lst_acu = []

for lst in accu_kfold_list:

    print("Index or number of variables used to build a model:",lst.index(max(lst))+1, "Max accuracy value :",max(lst) )

    lst_rfe.append(lst.index(max(lst))+1)

    lst_acu.append(max(lst))

    

print("List of 10 accuracies :",lst_acu)

print("List of indices or number of variables used :" ,lst_rfe)

mode_val = max(set(lst_rfe), key=lst_rfe.count)

print("The Final number of variables that i am using to build and report best model :",mode_val)

indices = [i for i, x in enumerate(lst_rfe) if x == mode_val]

print("The indices of most occuring k value :",indices)



# using operator.itemgetter() to get elements from list  

res_list = list(itemgetter(*indices)(lst_acu))

print("The accuracy values present at these indices :",res_list)

print("The Final mean cross validation accuracy for reporting purpose is :",mean(res_list))

#print("The number of variables used in the best model :",mode(lst_avg))
#Uninstalling XGBoost and downloading an earlier version as RFE is not compatible with the latest XGBoost.

!pip show xgboost
!pip uninstall xgboost --y
!pip install --upgrade xgboost==0.90
!pip show xgboost
# scikit-learn k-fold cross-validation



from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from itertools import compress

from numpy import array

from sklearn.model_selection import KFold, StratifiedKFold

from statistics import mode

from operator import itemgetter 

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



import warnings

warnings.filterwarnings('ignore')

warnings.simplefilter('ignore')



# Prepare for 10-fold cross validation

skf = StratifiedKFold(n_splits=10)

skf.get_n_splits(X_training, Y_training)

accu_kfold_list = []

counter_kfold =0

# enumerate splits

for train, val in skf.split(X_training, Y_training):

    counter_kfold+=1

    #print("KFold validation iteration :",counter_kfold)

    X_train, X_val = X_training.iloc[train], X_training.iloc[val]

    Y_train, Y_val = Y_training.iloc[train], Y_training.iloc[val]





    # create a base classifier used to evaluate a subset of attributes

    accu_list = []

    for num_of_vars in range(1,len(data_kaggle_feature1.columns)):

        

        # XG Boost Algorithm 

        XGB = XGBClassifier(learning_rate = 0.35, n_estimator = 200, silent = True, verbosity = 0)



        # create the RFE model 

        rfe = RFE(XGB, num_of_vars)

        rfe = rfe.fit(X_train, Y_train)



        

        # summarize the selection of the attributes

        #print("Number of variables used :",num_of_vars)

        col_list = data_kaggle_feature1.columns.to_list()[:-1]

        #print(col_list)

        col_bool = rfe.support_

        #print(col_bool)

        col_list = list(compress(col_list, col_bool))  

        #print("Variables used in building this classifier :",col_list)



        # Training the model - Fitting

        model_xgb = XGB.fit(X_train.filter(col_list), Y_train)

        #model_random = R_forest.fit(X_train.filter(col_list), Y_train)

        

        # Predictions

        #pred_random = model_random.predict(X_val.filter(col_list))

        pred_xgb = model_xgb.predict(X_val.filter(col_list))



        #accuracy = accuracy_score(Y_val, pred_random)

        accuracy = accuracy_score(Y_val, pred_xgb)

        #print ("The accuracy of XGBoost model is : ",accuracy)



        #Adding the accuracy to list

        accu_list.append(accuracy)



    #print(accu_list)

    accu_kfold_list.append(accu_list)





#List of all accuracies

#print(accu_kfold_list)



#Creating empty lists to calculate which variable configuration produced maximum acccuracy

lst_rfe = []

lst_acu = []

for lst in accu_kfold_list:

    print("Index or number of variables used to build a model:",lst.index(max(lst))+1, "Max accuracy value :",max(lst) )

    lst_rfe.append(lst.index(max(lst))+1)

    lst_acu.append(max(lst))

    

print("List of 10 accuracies :",lst_acu)

print("List of indices or number of variables used :" ,lst_rfe)

mode_val = max(set(lst_rfe), key=lst_rfe.count)

print("The Final number of variables that i am using to build and report best model :",mode_val)

indices = [i for i, x in enumerate(lst_rfe) if x == mode_val]

print("The indices of most occuring k value :",indices)



# using operator.itemgetter() to get elements from list  

res_list = list(itemgetter(*indices)(lst_acu))

print("The accuracy values present at these indices :",res_list)

print("The Final mean cross validation accuracy for reporting purpose is :",mean(res_list))

#print("The number of variables used in the best model :",mode(lst_avg))
data_kaggle[data_kaggle.workclass == 0]
#One hot Encoder - Encoding or feature engineering for categorical (non-ordinal) variables

#from sklearn.preprocessing import LabelEncoder

#Creating a copy of my dataframe

data_kaggle_ohe = data_kaggle.copy()

categorical_features_ohe = ['workclass','marital.status','occupation','relationship','race','sex','native.country']



label_encoder_feat = {}

new_df = pd.DataFrame() #creates a new dataframe that's empty

df_temp = pd.get_dummies(data_kaggle_ohe['workclass'], prefix='workclass')

for i, feature in enumerate(categorical_features_ohe[1:]):

    df_temp1 = pd.get_dummies(data_kaggle_ohe[feature], prefix=feature)

    df_temp = pd.concat([df_temp, df_temp1], axis=1)

    #label_encoder_feat[feature] = OneHotEncoder()

    #data_kaggle_ohe[feature] = label_encoder_feat[feature].fit_transform(data_kaggle_ohe[feature])



#data_kaggle_ohe.head()

ohe_columns_list = df_temp.columns.to_list()

print(ohe_columns_list)

#Changuing datatypes of all columns thus generated - from uint8 to int

df_temp = df_temp.astype(int)



data_kaggle_ohe = data_kaggle_ohe.drop(categorical_features_ohe, axis=1)

data_kaggle_ohe = pd.concat([data_kaggle_ohe,df_temp], axis=1)

data_kaggle_ohe.head(5)
# Feature Selection with Univariate Statistical Tests - for Categorical inputs

#List of columns that i need to hot encode

names = ohe_columns_list



X= data_kaggle_ohe[names]

Y= data_kaggle_ohe['income'].astype('int')

# Performing feature extraction using Chi-square method

test = SelectKBest(score_func=chi2, k=3)

fit = test.fit(X, Y)

# summarize scores

feature_list_cat = fit.scores_

np.set_printoptions(formatter={'float_kind':'{:f}'.format})



print("scores :",fit.scores_)

print("col names :",names)
#Creating a dictionary and dataframe woith scores and col names

dict_temp = {'scores': fit.scores_, 'cols': names}

df_scores = pd.DataFrame(dict_temp)

print ("old shape :",df_scores.shape)

#Limiting the number of columns for next step by only picking scores higher than 100

df_scores = df_scores[df_scores.scores >= 450]

print ("New shape :",df_scores.shape)

# a bar plot 

df_scores.plot(kind='bar',x='cols',y='scores',color='red')

plt.xticks(rotation=70)

plt.show()

data_kaggle_ohe.columns.to_list()
#Getting the name of  variables in a list

#These variables were identified using ANOVA analysis of numeric variables above

list1 = ['age', 'capital.flow','hours.per.week'] 

#This is the list of newly created variables using SelectKBest and One Hot ENcoding

list2 = df_scores.cols.to_list() 

#List of variables present in our dataset

list3 = data_kaggle_ohe.columns.to_list()



#Getting all this together

list4 = ['income','education.num']

for items in list3:

    if items in list1:

        list4.append(items)

    if items in list2:

        list4.append(items)

#Final list of variables for running RFE and Logistic regression

print("Final list of variables for building my model :",list4)

#Filtering dataset accordingly

print("Shape of final dataset :",data_kaggle_ohe[list4].shape)

data_kaggle_ohe[list4].head(10)

print(list4)

data_kaggle_ohe[list4[1:]] #.iloc[:,:]
#Creating Training(25,000)+Validation(15,000) and Test(10,000) sttartefied dataset - a split of 50-30-20% respectively

#First splitting dataset of 50,000 instances into training (80%) and test (20%)

from sklearn.model_selection import train_test_split

X_training, X_test, Y_training, Y_test = train_test_split(data_kaggle_ohe[list4[1:]], data_kaggle_ohe.income,

                                                    stratify=data_kaggle_ohe.income, 

                                                    test_size=0.10)





Y_training, Y_test =  Y_training.astype(int), Y_test.astype(int)

print("Shape of train split :",X_training.shape,Y_training.shape)

print("Shape of test split :",X_test.shape,Y_test.shape)
X_training.columns.to_list() #18 variables
# scikit-learn k-fold cross-validation

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from itertools import compress

from numpy import array

from sklearn.model_selection import KFold, StratifiedKFold

from statistics import mode

from operator import itemgetter 

from sklearn.ensemble import GradientBoostingClassifier



# Prepare for 10-fold cross validation

skf = StratifiedKFold(n_splits=10)

skf.get_n_splits(X_training, Y_training)

accu_kfold_list = []

counter_kfold =0

# enumerate splits

for train, val in skf.split(X_training, Y_training):

    counter_kfold+=1

    #print("KFold validation iteration :",counter_kfold)

    X_train, X_val = X_training.iloc[train], X_training.iloc[val]

    Y_train, Y_val = Y_training.iloc[train], Y_training.iloc[val]





    # create a base classifier used to evaluate a subset of attributes

    accu_list = []

    for num_of_vars in range(1,len(X_training.columns)):

        

        #Logistic Regression

        logreg = LogisticRegression()

        # create the RFE model and select 3 attributes

        rfe = RFE(logreg, num_of_vars)

        rfe = rfe.fit(X_train, Y_train)



        # summarize the selection of the attributes

        #print("Number of variables used :",num_of_vars)

        col_list = X_training.columns.to_list()[:-1]

        #print(col_list)

        col_bool = rfe.support_

        #print(col_bool)

        col_list = list(compress(col_list, col_bool))  

        #print("Variables used in building this classifier :",col_list)



        # fit

        logreg.fit(X_train.filter(col_list), Y_train)



        # predict

        Y_pred = logreg.predict(X_val.filter(col_list))



        accuracy = accuracy_score(Y_val, Y_pred)

        #print('LogReg %s' % accuracy)



        #Adding the accuracy to list

        accu_list.append(accuracy)

 

    #print(accu_list)

    accu_kfold_list.append(accu_list)





#List of all accuracies

#print(accu_kfold_list)



#Creating empty lists to calculate which variable configuration produced maximum acccuracy

lst_rfe = []

lst_acu = []

for lst in accu_kfold_list:

    print("Index or number of variables used to build a model:",lst.index(max(lst))+1, "Max accuracy value :",max(lst) )

    lst_rfe.append(lst.index(max(lst))+1)

    lst_acu.append(max(lst))

    

print("List of 10 accuracies :",lst_acu)

print("List of indices or number of variables used :" ,lst_rfe)  

mode_val = max(set(lst_rfe), key=lst_rfe.count)

print("The Final number of variables that i am using to build and report best model :",mode_val)

indices = [i for i, x in enumerate(lst_rfe) if x == mode_val]

print("The indices of most occuring k value :",indices)



# using operator.itemgetter() to get elements from list  

res_list = list(itemgetter(*indices)(lst_acu))

print("The accuracy values present at these indices :",res_list)

print("The Final mean cross validation accuracy for reporting purpose is :",mean(res_list))

#print("The number of variables used in the best model :",mode(lst_avg))
# scikit-learn k-fold cross-validation

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from itertools import compress

from numpy import array

from sklearn.model_selection import KFold, StratifiedKFold

from statistics import mode

from operator import itemgetter 

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



import warnings

warnings.filterwarnings('ignore')

warnings.simplefilter('ignore')



# Prepare for 10-fold cross validation

skf = StratifiedKFold(n_splits=10)

skf.get_n_splits(X_training, Y_training)

accu_kfold_list = []

counter_kfold =0

# enumerate splits

for train, val in skf.split(X_training, Y_training):

    counter_kfold+=1

    #print("KFold validation iteration :",counter_kfold)

    X_train, X_val = X_training.iloc[train], X_training.iloc[val]

    Y_train, Y_val = Y_training.iloc[train], Y_training.iloc[val]





    # create a base classifier used to evaluate a subset of attributes

    accu_list = []

    for num_of_vars in range(1,len(X_training.columns)):

        

        # XG Boost Algorithm 

        XGB = XGBClassifier(learning_rate = 0.35, n_estimator = 200, silent = True, verbosity = 0)



        # create the RFE model 

        rfe = RFE(XGB, num_of_vars)

        rfe = rfe.fit(X_train, Y_train)



        

        # summarize the selection of the attributes

        #print("Number of variables used :",num_of_vars)

        col_list = X_training.columns.to_list()[:-1]

        #print(col_list)

        col_bool = rfe.support_

        #print(col_bool)

        col_list = list(compress(col_list, col_bool))  

        #print("Variables used in building this classifier :",col_list)



        # Training the model - Fitting

        model_xgb = XGB.fit(X_train.filter(col_list), Y_train)

        #model_random = R_forest.fit(X_train.filter(col_list), Y_train)

        

        # Predictions

        #pred_random = model_random.predict(X_val.filter(col_list))

        pred_xgb = model_xgb.predict(X_val.filter(col_list))



        #accuracy = accuracy_score(Y_val, pred_random)

        accuracy = accuracy_score(Y_val, pred_xgb)

        #print ("The accuracy of XGBoost model is : ",accuracy)



        #Adding the accuracy to list

        accu_list.append(accuracy)



    #print(accu_list)

    accu_kfold_list.append(accu_list)





#List of all accuracies

#print(accu_kfold_list)



#Creating empty lists to calculate which variable configuration produced maximum acccuracy

lst_rfe = []

lst_acu = []

for lst in accu_kfold_list:

    print("Index or number of variables used to build a model:",lst.index(max(lst))+1, "Max accuracy value :",max(lst) )

    lst_rfe.append(lst.index(max(lst))+1)

    lst_acu.append(max(lst))

    

print("List of 10 accuracies :",lst_acu)

print("List of indices or number of variables used :" ,lst_rfe)

mode_val = max(set(lst_rfe), key=lst_rfe.count)

print("The Final number of variables that i am using to build and report best model :",mode_val)

indices = [i for i, x in enumerate(lst_rfe) if x == mode_val]

print("The indices of most occuring k value :",indices)



# using operator.itemgetter() to get elements from list  

res_list = list(itemgetter(*indices)(lst_acu))

print("The accuracy values present at these indices :",res_list)

print("The Final mean cross validation accuracy for reporting purpose is :",mean(res_list))

#print("The number of variables used in the best model :",mode(lst_avg))