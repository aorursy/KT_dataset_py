#Importing the data analysis libraries

import numpy as np # linear algebra

import pandas as pd # data processing



#Importing the visualization libraries

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

#Ensuring that we don't see any warnings while running the cells

import warnings

warnings.filterwarnings('ignore') 



#Importing the counter

from collections import Counter



#Importing sci-kit learn libraries that we will need for this project

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



train = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

test = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
train.sample(10)
train.describe(include="all")
print(pd.isnull(train).sum())
df = pd.concat(objs = [train, test], axis = 0).reset_index(drop=True)

df.describe(include="all")
print(pd.isnull(df).sum())
numerical_data = df.select_dtypes(include='number')

categorical_data = df.select_dtypes(exclude='number')
numerical_data.describe(include='all')
categorical_data.head()
sn = sns.heatmap(df[["Response",

                "Age",

                "Driving_License", 

                "Region_Code", 

                "Previously_Insured", 

                "Vehicle_Age", 

                "Vehicle_Damage", 

                "Annual_Premium",

                "Policy_Sales_Channel",

                "Vintage"]].corr(), cmap = 'coolwarm', annot = True)
#A function to visualize and determine the fraction of responses in each category for a certain feature

def bar_plot(feature):

    

    feature_categories = df[feature].sort_values().unique()

    for category in feature_categories:

        temp_series = df["Response"][df[feature] == category].value_counts(normalize = True)

        #This code is used to solve problem when there are no Responses for a category, which causes an error in runtime

        if temp_series.shape == (1,):

            temp_series = temp_series.append(pd.Series([0], index=[1]))

        elif temp_series.shape == (0,):

            continue

        print("Percentage of individuals having {}: {}, who got the insurance: {:.2f} %".format(feature, category, temp_series[1]*100))

    #visualize

    sns.barplot(x = df[feature],y = df["Response"],  data = df).set_title('Fraction Who Got Insurance With Respect To {}'.format(feature))
bar_plot("Gender")
bar_plot("Vehicle_Age")
bar_plot("Vehicle_Damage")
bar_plot("Driving_License")
bar_plot("Previously_Insured")
sn = sns.heatmap(df[["Response",

                    "Age", 

                    "Region_Code",

                    "Vehicle_Age",  

                    "Policy_Sales_Channel",

                    "Vintage"]].corr(), cmap = 'coolwarm', annot = True)
# A function that takes in a feature and returns the histogram

def histograms(feature):

    fig = px.histogram(

        train, 

        feature, 

        color='Response',

        nbins=100, 

        title=('{} Vs Response'.format(feature)), 

        width=700,

        height=500

    )

    fig.show()
histograms("Age")
histograms("Vintage")
histograms("Region_Code")
histograms("Policy_Sales_Channel")
histograms("Annual_Premium")
df["Vehicle_Age_Encoded"] = df["Vehicle_Age"].map({"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2})
df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})
df["Vehicle_Damage_Encoded"] = df["Vehicle_Damage"].map({"No": 0, "Yes": 1})
df.head()
df = df.drop(["Vehicle_Age", "Vehicle_Damage", "Gender"], axis = 1)
customer_ID = pd.Series(df["id"], name = "CustomerId")

df = df.drop(["id", "Vintage"], axis = 1)
df.sample(5)
train = df[:train.shape[0]]

test = df[train.shape[0]:].drop(["Response"], axis = 1)
#StratifiedKFold aims to ensure each class is (approximately) equally represented across each test fold

k_fold = StratifiedKFold(n_splits=5)



X_train = train.drop(labels="Response", axis=1)

y_train = train["Response"]



# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)



# Creating objects of each classifier

LG_classifier = LogisticRegression(random_state=0)

SVC_classifier = SVC(kernel="rbf", random_state=0)

KNN_classifier = KNeighborsClassifier()

NB_classifier = GaussianNB()

DT_classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)

RF_classifier = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=0)



#putting the classifiers in a list so I can iterate over there results easily

insurance_classifiers = [LG_classifier]



#This dictionary is just to grad the name of each classifier

classifier_dict = {

    0: "Logistic Regression",

    1: "Support Vector Classfication",

    2: "K Nearest Neighbor Classification",

    3: "Naive bayes Classifier",

    4: "Decision Trees Classifier",

    5: "Random Forest Classifier",

}



insurance_results = pd.DataFrame({'Model': [],'Mean Accuracy': [], "Standard Deviation": []})



#Iterating over each classifier and getting the result

for i, classifier in enumerate(insurance_classifiers):

    classifier_scores = cross_val_score(classifier, X_train, y_train, cv=k_fold, n_jobs=2, scoring="accuracy")

    insurance_results = insurance_results.append(pd.DataFrame({"Model":[classifier_dict[i]], 

                                                           "Mean Accuracy": [classifier_scores.mean()],

                                                           "Standard Deviation": [classifier_scores.std()]}))
print (insurance_results.to_string(index=False))
# from sklearn.model_selection import GridSearchCV



# RF_classifier = RandomForestClassifier()





# ## Search grid for optimal parameters

# RF_paramgrid = {"max_depth": [None],

#                   "max_features": [1, 3, 10],

#                   "min_samples_split": [2, 3, 10],

#                   "min_samples_leaf": [1, 3, 10],

#                   "bootstrap": [False],

#                   "n_estimators" :[100,200,300],

#                   "criterion": ["entropy"]}





# RF_classifiergrid = GridSearchCV(RF_classifier, param_grid = RF_paramgrid, cv=k_fold, scoring="accuracy", n_jobs= -1, verbose=1)



# RF_classifiergrid.fit(X_train,y_train)



# RFC_optimum = RF_classifiergrid.best_estimator_



# # Best Accuracy Score

# RF_classifiergrid.best_score_
IDtest = customer_ID[train.shape[0]:].reset_index(drop = True)


X_train = train.drop(labels="Response", axis=1)

y_train = train["Response"]



# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(test)



LG_classifier.fit(X_train, y_train)



test_predictions = pd.Series(LG_classifier.predict(X_test).astype(int), name="Response")

insurance_results = pd.concat([IDtest, test_predictions], axis = 1)

insurance_results.to_csv('submission.csv', index=False)