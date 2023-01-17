import numpy as np 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = 40
# Let's load the data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
# We concat the two dataset in order to work on the missing values

full_dataset = pd.concat([train_data, test_data])

# Let's then have a look at the features

full_dataset.head()
from sklearn.base import BaseEstimator, TransformerMixin



class KnownCabinTransformer(BaseEstimator, TransformerMixin):

    """

    This first transformer is rather easy, goal is to create

    a new column, 'KnownCabin' where value is 0 when we don't

    know the cabin location, and 1 when we do.

    For this notebook, this is the only thing I do with cabin,

    but I'm sure there is plenty more to do with this information.

    """

    def __init__(self):

        print("- KnownCabin transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X.Cabin.fillna(0, inplace=True)

        X["KnownCabin"] = 0

        X.loc[X.Cabin != 0, 'KnownCabin'] = 1

        

        print("- KnownCabin transformer applied -")

             

        return X

            
class TitleTransformer(BaseEstimator, TransformerMixin):

    """ 

    A way more interesting Transformer, when I extract the Title 

    of every passenger thanks to a regular expression, and create

    dummies out of them. Those Titles also gonna be useful for 

    other features later on.

    """

    def __init__(self):

        print("- Title transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X['Title'] = X.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

        

        #Different variations of nobility/professions, all called 'Rare'

        X['Title'] = X['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                         'Don', 'Dr', 'Major', 'Rev', 

                                         'Sir', 'Jonkheer', 'Dona'], 

                                         'Rare')

        

        #Different (French) variations of Miss and Mrs

        X['Title'] = X['Title'].replace(['Mlle', 'Ms'], 'Miss')

        X['Title'] = X['Title'].replace('Mme', 'Mrs')

        

        #If a title is not treated, I count it as rare

        X['Title'] = X['Title'].fillna('Rare')

        

        title_dummies = pd.get_dummies(X['Title'], prefix="Title")

        

        X = pd.concat([X, title_dummies], axis=1)

        

        # Drop of Title column for redundancy

        X = X.drop('Title', axis=1)

        

        

        print("- Title transformer applied -")

             

        return X
class MissingFareTransformer(BaseEstimator, TransformerMixin):

    """

    Simple transformer to handle missing values in Fare. It looks

    at the class of the passenger before giving the median fare 

    associated with this class.

    It's not optimal as you would also like to look at the number 

    of people under the same ticket, but it's a beginning.    

    """

    def __init__(self):

        print("- MissingFare transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X.loc[(X.Pclass == 1) & (X.Fare.isnull()), 'Fare'] = X.loc[X.Pclass == 1]["Fare"].median()

        X.loc[(X.Pclass == 2) & (X.Fare.isnull()), 'Fare'] = X.loc[X.Pclass == 2]["Fare"].median()

        X.loc[(X.Pclass == 3) & (X.Fare.isnull()), 'Fare'] = X.loc[X.Pclass == 3]["Fare"].median()

        

        print("- MissingFare transformer applied -")

        

        return X

    



class EmbarkedTransformer(BaseEstimator, TransformerMixin):

    """

    An other simple transformer, giving to the missing values in

    Embarked the most common value, S in that case. We also use

    this transformer to create dummies of Embarked.

    """

    def __init__(self):

        print("- MissingEmbarked transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        self.most_frequent = X["Embarked"].value_counts().index[0]

        X["Embarked"].fillna(self.most_frequent, inplace=True)

        

        embarked_dummies = pd.get_dummies(X['Embarked'], prefix="Embarked")

        

        X = pd.concat([X, embarked_dummies], axis=1)

        

        # Drop of Embarked column for redundancy

        X = X.drop('Embarked', axis=1)

        

        print("- MissingEmbarked transformer applied -")

        

        return X

    



class HypotheticalMissingsTransformer(BaseEstimator, TransformerMixin):

    """

    This one is actually useless, but future-proof !! If the test set

    changes again, this transformer will deal with new missing values 

    in features without missing values at the time of this notebook.

    The only column I don't deal with is the Ticket, being a tad to

    difficult to generate random Ticket or finding families that would

    share the same number.

    """

    def __init__(self):

        print("- HypotheticalMissings transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        #Pclass is handled by frequency

        self.most_frequent_class = X["Pclass"].value_counts().index[0]

        

        self.unknown_name = "unknown"

        

        # Assuming people with NaN in SibSp and Parch got no family with them

        self.horizontal_family = 0

        self.vertical_family = 0     

        

        X["Pclass"].fillna(self.most_frequent_class, inplace=True)

        X["Name"].fillna(self.unknown_name, inplace=True)

        

        X["SibSp"].fillna(self.horizontal_family, inplace=True)

        X["Parch"].fillna(self.vertical_family, inplace=True)

        

        # We go a little further with sex, kinda important in our models

        # Mr., Master. and Rare are men, others are female

        X.loc[(X.Sex.isnull()) & (X.Title_Master == 1), "Sex"] == "male"

        X.loc[(X.Sex.isnull()) & (X.Title_Mr == 1), "Sex"] == "male"

        # We assume 'rare' title holders are more likely to be male than female

        X.loc[(X.Sex.isnull()) & (X.Title_Rare == 1), "Sex"] == "male"

        

        X.loc[(X.Sex.isnull()) & (X.Title_Miss == 1), "Sex"] == "female"

        X.loc[(X.Sex.isnull()) & (X.Title_Mrs == 1), "Sex"] == "female"

        

        print("- HypotheticalMissings transformer applied -")

        

        return X



    

class GenderTransformer(BaseEstimator, TransformerMixin):

    """

    This transformer could have been a simple use of dummies,

    but at this point I was full-on in the use of transformers,

    transformers are cool, Michael Bay approves !

    """

    def __init__(self):

        print("- Gender transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

        

        print("- Gender transformer applied -")

        

        return X





class PclassTransformer(BaseEstimator, TransformerMixin):

    """

    Cf GenderTransformer commentaries, transformers rule!

    """

    def __init__(self):

        print("- Pclass transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        pclass_dummies = pd.get_dummies(X['Pclass'], prefix='Pclass')

        

        X = pd.concat([X, pclass_dummies], axis=1)

        

        X = X.drop('Pclass', axis=1)

        

        print("- Pclass transformer applied -")

        

        return X

    



class FamilySizeTransformer(BaseEstimator, TransformerMixin):

    """

    A simple feature created thanks to other existing features.

    FamilySize is used by many other notebooks and yield some

    decent results.

    """

    def __init__(self):

        print("- FamilySize transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1

        

        print("- FamilySize transformer applied -")

        

        return X
class AgeStatusTransformer(BaseEstimator, TransformerMixin):

    """

    I wanted to distinguish people from whom we know for sure their age,

    those from whom the age is estimated (floating value over 1 yo), and

    those from whom we guessed the age (missing values at the start of the

    exercice)

    """

    def __init__(self):

        print("- AgeStatus transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        X["AgeStatus"] = "known"

        #Toddlers under 1 have their age displayed in float, it's not a estimation

        X.loc[(X["Age"] > 1) & ((X["Age"]*2)%2 != 0), "AgeStatus"] = "estimated"

        X.loc[X["Age"].isnull(), "AgeStatus"] = "guessed"

        

        age_status_dummies = pd.get_dummies(X['AgeStatus'], prefix="AgeStatus")

        

        X = pd.concat([X, age_status_dummies], axis=1)

        

        # Drop of AgeStatus column for redundancy

        X = X.drop('AgeStatus', axis=1)

        

        print("- AgeStatus transformer applied -")

             

        return X

            
import category_encoders as ce



class TicketGroupingTransformer(BaseEstimator, TransformerMixin):

    """

    I apply a category encoder on the Ticket, if 6 people share the

    same ticket, the value of the new column will be 6.

    """

    def __init__(self):

        print("- TicketGrouping transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        cat_feature = ['Ticket']

        count_enc = ce.CountEncoder(cols=cat_feature)

        count_enc.fit(X[cat_feature])

        grouping_col = count_enc.transform(X[cat_feature]).add_suffix("_grouping")

        X = pd.concat([X, grouping_col], axis=1)

        

        print("- TicketGrouping transformer applied -")

        

        return X
class AgeGuessingTransformer(BaseEstimator, TransformerMixin):

    """

    The transformer that guess the ages way more precisely than mean or

    median, and I made the previous change in part two. Every change is

    explained but it is rather intuitive.

    

    I never apply hard-coded value of course, I just use the median of

    a more precise subgroup.

    """

    def __init__(self):

        print("- AgeGuessing transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        # Every 'Master' is a child, we apply the median of all Master (4yo)

        X.loc[(X.Age.isnull() == True) & (X.Title_Master == 1), "Age"] = X.loc[X.Title_Master == 1]["Age"].median()

        

        # Every Miss with at least a sibling (no spouse obviously), is more 

        # likely to be a child too (it's not sure, just more likely) (13yo)

        X.loc[(X.Age.isnull() == True) & (X.SibSp > 0) & (X.Title_Miss == 1), "Age"] = X.loc[(X.Title_Miss == 1) & (X.SibSp > 0)]["Age"].median()

        

        # People alone are way more likely to be adult (28yo)

        X.loc[(X.Age.isnull() == True) & (X.Ticket_grouping == 1), "Age"] = X.loc[(X.Ticket_grouping == 1)]["Age"].median()

        

        # People with more than 1 sibling/spouse are more likely to be children (13yo)

        X.loc[(X.Age.isnull() == True) & (X.SibSp > 1), "Age"] = X.loc[(X.Age.isnull() == False) & (X.SibSp > 1)]["Age"].median()

        

        # The rest is more likely to be adults (27yo)

        X.loc[(X.Age.isnull()), "Age"] = X.loc[(X.Age.isnull() == False) & (X.Ticket_grouping > 1)]["Age"].median()

        

        

        print("- AgeGuessing transformer applied -")

        

        return X
class AgeBinningTransformer(BaseEstimator, TransformerMixin):

    """

    I use a binning technique to create 6 categories of age,

    a better solution would be to look at the distribution of

    age before applying the bins, I was just lazy.

    """

    def __init__(self):

        print("- AgeBinning transformer initiated -")

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        self.bins = [0, 2, 16, 25, 40, 50, np.inf]

        self.age_cat = ["0_2", "2_16", "16_25", "25_40", "30_50", "50+"]

        

        X["AgeBins"] = pd.cut(X["Age"], bins=self.bins, labels=self.age_cat)

        

        age_bins_dummies = pd.get_dummies(X['AgeBins'], prefix="AgeBins")

        

        X = pd.concat([X, age_bins_dummies], axis=1)  

        

        print("- AgeBinning transformer applied -")

        

        return X
from sklearn.pipeline import Pipeline



custom_pipeline = Pipeline([

        ("cabin_trans", KnownCabinTransformer()),

        ("title_trans", TitleTransformer()),

        ("hypo_missing_trans", HypotheticalMissingsTransformer()),

        ("missing_fare_trans", MissingFareTransformer()),

        ("embarked_trans", EmbarkedTransformer()),

        ("gender_trans", GenderTransformer()),

        ("class_trans", PclassTransformer()),

        ("family_size_trans", FamilySizeTransformer()),

        ("ticket_grp_trans", TicketGroupingTransformer()),

        ("age_status_trans", AgeStatusTransformer()),

        ("age_guessing_trans", AgeGuessingTransformer()),

        ("age_binning_trans", AgeBinningTransformer()),

    ])



full_dataset = custom_pipeline.fit_transform(full_dataset)
# LOOK AT THIS, SUCH A BEAUTY !!!

full_dataset.head(3)
# I set up the column as null, staying this way for people in group of 2 or less

full_dataset["TS_Tragedy"] = "null"

i = 0

unique_tickets_subset = full_dataset.loc[full_dataset["Ticket_grouping"] > 2]["Ticket"].unique()

while i < len(unique_tickets_subset):

    subset = full_dataset.loc[full_dataset.Ticket == unique_tickets_subset[i]]

    try:

        # I chose to have the number of people in training set as denominator rather than total on ticket

        ratio = len(subset.loc[subset["Survived"] == 0]) / len(subset.loc[subset.Survived.isnull() == False])

        if ratio > 0.65 :

            full_dataset.loc[full_dataset.Ticket == unique_tickets_subset[i], "TS_Tragedy"] = "yes"

        else:

            full_dataset.loc[full_dataset.Ticket == unique_tickets_subset[i], "TS_Tragedy"] = "no"

        i += 1

    

    #ZeroDivisionError caused by a full group in test set, I pass on it

    except ZeroDivisionError:

        i += 1

        

TST_dummies = pd.get_dummies(full_dataset['TS_Tragedy'], prefix="TST")

        

full_dataset = pd.concat([full_dataset, TST_dummies], axis=1)

        

full_dataset = full_dataset.drop('TS_Tragedy', axis=1)



full_dataset.head(3)

# We reindex the full dataset, before dropping the PassengerId col later on

full_dataset.set_index(full_dataset.PassengerId, verify_integrity = True, inplace=True)

full_dataset.tail()
# split back into train and test



X_train = full_dataset.loc[full_dataset.Survived.isnull() == False]

y_train = X_train["Survived"].astype(int)

X_test = full_dataset.loc[full_dataset.Survived.isnull()]



### Dropping non used features (and target Survived)

useless_features = ["PassengerId", "Survived", "Name", "Age", "Ticket", "Fare", "Cabin", "AgeBins"]



X_train.drop(useless_features, axis=1, inplace=True)

X_test.drop(useless_features, axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[["SibSp","Parch","FamilySize","Ticket_grouping"]]), columns=["sc_SibSp","sc_Parch","sc_FamilySize","sc_Ticket_grouping"], index=X_train.index)

X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test[["SibSp","Parch","FamilySize","Ticket_grouping"]]), columns=["sc_SibSp","sc_Parch","sc_FamilySize","sc_Ticket_grouping"], index=X_test.index)



X_train = pd.concat([X_train, X_train_scaled], axis=1)

X_train.drop(["SibSp","Parch","FamilySize","Ticket_grouping"], axis=1, inplace=True)

X_test = pd.concat([X_test, X_test_scaled], axis=1)

X_test.drop(["SibSp","Parch","FamilySize","Ticket_grouping"], axis=1, inplace=True)



X_train.head(3)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
# Various imports

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import xgboost as xgb

import lightgbm as lgb

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
# KNeighbors example



param_grid = [

    {'n_neighbors':[2,3,4,5,6,7,8], 'weights':['uniform', 'distance'], 'n_jobs':[-1] }

]



neigh = KNeighborsClassifier()



grid_search = GridSearchCV(neigh, param_grid, cv=5, scoring='accuracy')



grid_search.fit(X_train, y_train)



grid_search.best_params_
neigh = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform')

neigh.fit(X_train, y_train)

scores = cross_val_score(neigh, X_train, y_train, scoring="accuracy", cv=10)

print(scores.mean())
"""

# Logistic Regression



param_grid = [

    {'penalty':['l1','l2','elacticnet','none'], 'C':[0.001, 0.01, 0.1,1, 10, 100, 1000], 'n_jobs':[-1] }

]



log_reg = LogisticRegression()



grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')



grid_search.fit(X_train, y_train)



grid_search.best_params_

"""

log_reg = LogisticRegression(C = 0.001, penalty='none')

log_reg.fit(X_train, y_train)

scores = cross_val_score(log_reg, X_train, y_train, scoring="accuracy", cv=10)

print("Logistic Regression mean score : ", scores.mean())



"""

# Random Forest



param_grid = [

    {'n_estimators':[100,200,300,500,700,1000], 'max_depth':[2,3,4,5,6,7,8],

     'min_samples_split':[1,2,3,4,5], 'min_samples_leaf':[1,2,3,4] ,'n_jobs':[-1] }

]



forest_clf = RandomForestClassifier()



grid_search = GridSearchCV(forest_clf, param_grid, cv=5, scoring='accuracy')



grid_search.fit(X_train, y_train)



grid_search.best_params_

"""



forest_clf = RandomForestClassifier(max_depth = 6, min_samples_leaf=1, min_samples_split=2, n_estimators=100)

forest_clf.fit(X_train, y_train)

scores = cross_val_score(forest_clf, X_train, y_train, scoring="accuracy", cv=10)

print("Random Forest mean score : ", scores.mean())



"""

# SVC



param_grid = [

    {'kernel':['linear','poly','rbf','sigmoid'], 'C':[0.001, 0.01, 0.1,1, 10, 100, 1000], 'degree':[2,3,4], 'gamma':['auto']}

]



svc = SVC()



grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')



grid_search.fit(X_train, y_train)



grid_search.best_params_

"""



# Despite this 0.847 score, real submission is 0.794, not good enough

svc = SVC(C = 10, kernel='rbf', gamma='auto', probability=True)

svc.fit(X_train, y_train)

scores = cross_val_score(svc, X_train, y_train, scoring="accuracy", cv=10)

print("SVC mean score: ", scores.mean())



# GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.01)

gbc.fit(X_train, y_train)

scores = cross_val_score(gbc, X_train, y_train, scoring="accuracy", cv=10)

print("Gradient Boosting mean score : ", scores.mean())



# XGB

xgb_clf = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.01)

xgb_clf.fit(X_train, y_train)

scores = cross_val_score(xgb_clf, X_train, y_train, scoring="accuracy", cv=10)

print("XGB mean score : ", scores.mean())

# Voting ensembling



voting_clf = VotingClassifier(estimators=[('gbc', gbc), ('xgb', xgb_clf), ('forest', forest_clf), ('svc', svc), ('lrc', log_reg), ('knc', neigh)],

                             voting='hard')



voting_clf.fit(X_train, y_train)



scores = cross_val_score(voting_clf, X_train, y_train, scoring="accuracy", cv=10)



print(scores.mean())
preds = voting_clf.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': preds})

output.to_csv("submission.csv", index=False)

output.head()