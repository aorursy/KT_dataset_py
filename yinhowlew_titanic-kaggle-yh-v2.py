import numpy as np

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# data will be used for training and validation

# test will be used for final evaluation

data = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

data.head()
data.describe(include="all")
test.describe(include="all")
print("train dataset has " + str(len(data)) + " passengers")

print("test dataset has " + str(len(test)) + " passengers")
# exploring our feature data types

data.dtypes
# see how many feature and label data are missing for both train and test

pd.DataFrame({"train": data.isna().sum(),

              "test": test.isna().sum()})
sns.barplot(x="Sex", y="Survived", data=data);
print("Percentage of females who survived:", str(round(data["Survived"][data["Sex"] == 'female'].value_counts(normalize = True)[1]*100, 2)),"%")



print("Percentage of males who survived:", str(round(data["Survived"][data["Sex"] == 'male'].value_counts(normalize = True)[1]*100, 2)),"%")
senior = data["Age"][data["Age"] > 60].count()

adult = data["Age"][data["Age"] <= 60][data["Age"] > 40].count()

youngadult = data["Age"][data["Age"] <= 40][data["Age"] > 20].count()

teenager = data["Age"][data["Age"] <= 20][data["Age"] > 12].count()

child = data["Age"][data["Age"] <= 12][data["Age"] > 4].count()

toddler = data["Age"][data["Age"] <= 4][data["Age"] > 1].count()

baby = data["Age"][data["Age"] <= 1][data["Age"] >= 0].count()

missing = data["Age"].isna().sum()

total = senior+adult+youngadult+teenager+child+toddler+baby+missing



print("Passengers by Age Group in train set")

print("senior:", senior)

print("adult:", adult)

print("youngadult:", youngadult)

print("teenager:", teenager)

print("child:", child)

print("toddler:", toddler)

print("baby:", baby)

print("missing:", missing)

print("total:", total)

print("Percentage of senior who survived:", str(round(data["Survived"][data["Age"] > 60].value_counts(normalize = True)[1]*100, 2)),"%")

print("Percentage of adult who survived:", str(round(data["Survived"][data["Age"] <= 60][data["Age"] > 40].value_counts(normalize = True)[1]*100, 2)),"%")

print("Percentage of youngadult who survived:", str(round(data["Survived"][data["Age"] <= 40][data["Age"] > 20].value_counts(normalize = True)[1]*100, 2)),"%")

print("Percentage of teen who survived:", str(round(data["Survived"][data["Age"] <= 20][data["Age"] > 12].value_counts(normalize = True)[1]*100, 2)),"%")

print("Percentage of child who survived:", str(round(data["Survived"][data["Age"] <= 12][data["Age"] > 4].value_counts(normalize = True)[1]*100, 2)),"%")

print("Percentage of toddler who survived:", str(round(data["Survived"][data["Age"] <= 4][data["Age"] > 1].value_counts(normalize = True)[1]*100, 2)),"%")

print("Percentage of baby who survived:", str(round(data["Survived"][data["Age"] <= 1][data["Age"] > 0].value_counts(normalize = True)[1]*100, 2)),"%")

print("Percentage of age-missing who survived:", str(round(data["Survived"][data["Age"].isna()].value_counts(normalize = True)[1]*100, 2)),"%")



sns.barplot(x="Pclass", y="Survived", data=data)



#print percentage of people by Pclass that survived

print("Percentage of Pclass = 1 who survived:", data["Survived"][data["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 2 who survived:", data["Survived"][data["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 3 who survived:", data["Survived"][data["Pclass"] == 3].value_counts(normalize = True)[1]*100)
data["Fare"].min(), data["Fare"].max(), data["Fare"].mean()
data["Fare"].plot.hist();
# grouping fare

high_fare = data["Fare"][data["Fare"] > 200].count()

mid_fare = data["Fare"][data["Fare"] <= 200][data["Fare"] > 50].count()

low_fare = data["Fare"][data["Fare"] <= 50][data["Fare"] > 25].count()

dirt_fare = data["Fare"][data["Fare"] <= 25].count()



# print("high: ", high_fare)

high_fare, mid_fare, low_fare, dirt_fare



# testing hypothesis: percentage of high-fare group who survived

print("Percentage of high-fare who survived:", data["Survived"][data["Fare"] > 200].value_counts(normalize = True)[1]*100)

print("Percentage of mid-fare who survived:", data["Survived"][data["Fare"] <= 200][data["Fare"] > 50].value_counts(normalize = True)[1]*100)

print("Percentage of low-fare who survived:", data["Survived"][data["Fare"] <= 50][data["Fare"] > 25].value_counts(normalize = True)[1]*100)

print("Percentage of dirt-fare who survived:", data["Survived"][data["Fare"] <= 25].value_counts(normalize = True)[1]*100)
# survival rate by no. of siblings

sns.barplot(x="SibSp", y="Survived", data=data);
sns.barplot(x="Parch", y="Survived", data=data);
has_cabin = data["Cabin"][data["Cabin"] != None].count()

no_cabin = data["Cabin"].isna().sum()

has_cabin, no_cabin
# explore survival rate of those with recorded cabins vs without



print("Percentage of has_cabin who survived:", data["Survived"][data["Cabin"] != None].value_counts(normalize = True)[1]*100)



print("Percentage of no_cabin who survived:", data["Survived"][data["Cabin"].isna()].value_counts(normalize = True)[1]*100)
# explore survival rate by cabin (A to G)

# we are filling up missing with "Z" since str.contains doesn't work on missing value

data["Cabin"].fillna("Z", inplace=True)
cabin_class = ["A", "B", "C", "D", "E", "F", "G", "Z"]



for cabin in cabin_class:

    print("% of cabin class", cabin, "who survived: ", 

          str(round(data["Survived"][data["Cabin"].str.contains(cabin)].value_counts(normalize = True)[1]*100)), "%",

          "out of ", data["Cabin"].str.contains(cabin).sum(), "passengers")
print("Number of people embarking in Southampton (S):", data["Embarked"][data["Embarked"] == "S"].count())



print("Number of people embarking in Cherbourg (C):", data["Embarked"][data["Embarked"] == "C"].count())



print("Number of people embarking in Queenstown (Q):", data["Embarked"][data["Embarked"] == "Q"].count())
print("Percentage of S who survived:", data["Survived"][data["Embarked"] == "S"].value_counts(normalize = True)[1]*100)

print("Percentage of C who survived:", data["Survived"][data["Embarked"] == "C"].value_counts(normalize = True)[1]*100)

print("Percentage of Q who survived:", data["Survived"][data["Embarked"] == "Q"].value_counts(normalize = True)[1]*100)
#create a combined group of both datasets

combine = [data, test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(data['Title'], data['Sex'])
# test our hypothesis if certain titles have higher survival rate

# note that sample size is limited, so we will only be testing a few



print("Master", data["Survived"][data["Title"] == "Master"].value_counts(normalize = True)[1]*100)

print("Miss", data["Survived"][data["Title"] == "Miss"].value_counts(normalize = True)[1]*100)

print("Mrs", data["Survived"][data["Title"] == "Mrs"].value_counts(normalize = True)[1]*100)

print("Dr", data["Survived"][data["Title"] == "Dr"].value_counts(normalize = True)[1]*100)
#get a list of the features within the dataset

print(data.columns)

print(test.columns)
# Split into X and y

X = data.drop("Survived", axis=1)

y = data["Survived"]
# writing our function



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



def transform_data(data):

    # create new feature: HasCabin

    data["Cabin"].fillna("Z", inplace=True) # we previously filled missing with Z on train, but haven't done for test

    data["HasCabin"] = data["Cabin"].str.contains("Z").astype('int')

    

    # create new feature: AgeGroup

    data["Age"] = data["Age"].fillna(-0.5)

    bins = [-1, 0, 1, 4, 12, 20, 40, 60, np.inf]

    labels = ['AgeMissing', 'Baby', 'Toddler', 'Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']

    data['AgeGroup'] = pd.cut(data["Age"], bins, labels = labels)    



    # create new feature: FareGroup

    data["Fare"] = data["Fare"].fillna(-0.5)

    bins_2 = [-1, 0, 25, 50, 200, np.inf]

    labels_2 = ['FareMissing', 'DirtFare', 'LowFare', 'MidFare', 'HighFare']

    data['FareGroup'] = pd.cut(data["Fare"], bins_2, labels = labels_2)     

    

    # remove "PassengerId", "Name", "Ticket", "Title", "Cabin", "Age", "Fare"

    data = data.drop("PassengerId", axis=1)

    data = data.drop("Name", axis=1)

    data = data.drop("Ticket", axis=1)

    data = data.drop("Title", axis=1)

    data = data.drop("Cabin", axis=1)

    data = data.drop("Age", axis=1)

    data = data.drop("Fare", axis=1)                                                            

    

    # fill na with pandas

    # HasCabin, AgeGroup, and FareGroup already filled above, so only need to fill Embarked

    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

    

    # transform "Sex", "Embark", "AgeGroup", "FareGroup", "Pclass", "HasCabin"

    # One Hot Encoding transforms categories into different feature columns of 1 and 0

    # note even though "Pclass" data is in numbers (1,2,3), they are Categorical features (instead of numerical like Age) hence we need to encode it

    one_hot = OneHotEncoder()

    transformer = ColumnTransformer([("one_hot",

                                      one_hot,

                                      ["Sex", "Embarked", "AgeGroup", "FareGroup", "Pclass", "HasCabin"])],

                                    remainder="passthrough")

    data = transformer.fit_transform(data)

    

    return data
# applying our function to transform X_train



X_tf = transform_data(X)

pd.DataFrame(X_tf)

# note this converts to numpy array, and not pd
# number of columns we should have

1+1+2+3+8+5+3+2
# check that number of rows is intact

len(X_tf), len(X)
# check if any missing value

pd.DataFrame(X_tf).isna().sum()
# Split data into train and validation sets

from sklearn.model_selection import train_test_split

np.random.seed(17)

X_train, X_val, y_train, y_val = train_test_split(X_tf, y, test_size=0.2)
# confirm splitting is done right

len(X_train), len(X_val), len(y_train), len(y_val)
# Creating a function to fit and score across the models



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Put models in a dictionary

models = {"Logistic Regression": LogisticRegression(),

          "KNN": KNeighborsClassifier(),

          "Random Forest": RandomForestClassifier(),

          "Gradient Boosting": GradientBoostingClassifier(),

          "SVC": SVC(),

          "SGD": SGDClassifier(),

          "Decision Tree": DecisionTreeClassifier()}





def fit_and_score(models, X_train, X_val, y_train, y_val):

    """

    Fits and evaluates given machine learning models.

    models : a dict of differetn Scikit-Learn machine learning models

    """

    # Set random seed

    np.random.seed(17)

    # Make a dictionary to keep model scores

    model_scores = {}

    # Loop through models

    for name, model in models.items():

        # Fit the model to the data

        model.fit(X_train, y_train)

        # Evaluate the model and append its score to model_scores

        model_scores[name] = model.score(X_val, y_val)

    return model_scores

# Executing our function, and we will see the "accuracy" score (using default scoring method of each model)

fit_and_score(models=models,

              X_train=X_train,

              X_val=X_val,

              y_train=y_train,

              y_val=y_val)
# creating a evaluation metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def evaluate_preds(y_true, y_preds):

    """

    perform evaluation comparison on y_true labels vs y_preds labels

    """

    accuracy = accuracy_score(y_true, y_preds)

    precision = precision_score(y_true, y_preds)

    recall = recall_score(y_true, y_preds)

    f1 = f1_score(y_true, y_preds)

    metric_dict = {"accuracy": round(accuracy, 2),

                    "precision": round(precision, 2),

                    "recall": round(recall, 2),

                    "f1": round(f1, 2)}

    print(f"Acc: {accuracy * 100:.2f}%")

    print(f"Precision: {precision:.2f}")

    print(f"Recall: {recall:.2f}")

    print(f"F1 score: {f1:.2f}")

    return metric_dict
# RandomForest



np.random.seed(19)

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_preds = rf.predict(X_val)  # ML's prediction using X_val



# evaluate using our evaluation function on validation set

rf_metrics = evaluate_preds(y_val, y_preds)  # compares y_preds with y_val/y_true

rf_metrics
# SVC



np.random.seed(19)

svc = SVC()

svc.fit(X_train, y_train)

y_preds = svc.predict(X_val)  # ML's prediction using X_val



# evaluate using our evaluation function on validation set

svc_metrics = evaluate_preds(y_val, y_preds)  # compares y_preds with y_val/y_true

svc_metrics
# Decision Tree



np.random.seed(19)

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

y_preds = decision_tree.predict(X_val)  # ML's prediction using X_val



# evaluate using our evaluation function on validation set

decision_tree_metrics = evaluate_preds(y_val, y_preds)  # compares y_preds with y_val/y_true

decision_tree_metrics

# Lets compare our different models with new metrics



compare_metrics = pd.DataFrame({"Random Forest": rf_metrics,

                                "SVC": svc_metrics,

                                "Decision Tree": decision_tree_metrics})

compare_metrics.plot.bar(figsize=(10,8));
# check available hyperparameters for SVC

svc.get_params()
# tuning hyperparameters by RandomSearchCV



from sklearn.model_selection import RandomizedSearchCV

grid = {"kernel": ["linear", "rbf", "poly"],

        "gamma": ["scale", "auto"],

        "degree": [0,1,2,3,4,5,6],

        "class_weight": ["balanced", None],

        "C": [100, 10, 1.0, 0.1, 0.001]}



np.random.seed(17)



svc = SVC()

rs_svc = RandomizedSearchCV(estimator=svc,

                            param_distributions=grid,  # what we defined above

                            n_iter=10, # number of combinations to try

                            cv=5,   # number of cross-validation split

                            verbose=2)

rs_svc.fit(X_train, y_train);
# checking out best parameters we find from tuning

rs_svc.best_params_
# evaluating our model tuned with RandomSearchCV

rs_y_preds = rs_svc.predict(X_val)



# evaluate predictions

rs_metrics = evaluate_preds(y_val, rs_y_preds)
# tuning hyperparameters by GridSearchCV

from sklearn.model_selection import GridSearchCV





grid_2 = {"kernel": ["linear", "rbf"],

        "gamma": ["scale"],

        "degree": [1,2,3],

        "class_weight": [None],

        "C": [100, 10, 1.0]}



np.random.seed(17)



svc = SVC()



# Setup GridSearchCV

gs_svc = GridSearchCV(estimator=svc,

                      param_grid=grid_2,

                      cv=5,

                      verbose=2)



# Fit the GSCV version of clf

gs_svc.fit(X_train, y_train);

gs_svc.best_params_
# evaluating our model tuned with GridSearchCV

gs_y_preds = gs_svc.predict(X_val)



# evaluate predictions

gs_metrics = evaluate_preds(y_val, gs_y_preds)
# Lets compare our different model metrics



compare_metrics = pd.DataFrame({"baseline SVC": svc_metrics,

                                "random search": rs_metrics,

                                "grid search": gs_metrics})

compare_metrics
# with bar graph

compare_metrics.plot.bar(figsize=(10,8));
# our raw test data

test.head()

# tranforming test data

test_tf = transform_data(test)

pd.DataFrame(test_tf)
# check no missing value

pd.DataFrame(test_tf).isna().sum()
# run our prediction with our rs_svc model



test_preds = rs_svc.predict(test_tf)

pd.DataFrame(test_preds)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], 

                          "Survived": test_preds})

submission
# check length



len(submission), len(test)
submission.to_csv("submission_v2.csv", index=False)