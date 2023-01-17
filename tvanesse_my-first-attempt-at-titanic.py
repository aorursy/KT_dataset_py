import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Machine Learning

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



train_df.head()
train_df.count()
train_df["Ticket"].nunique()
train_df.drop("Ticket", axis=1, inplace=True)

test_df.drop("Ticket", axis=1, inplace=True)

train_df.head(1)
age_dist = train_df["Age"].dropna()

ax = sns.distplot(age_dist, kde=False, color='k')

plt.title("Overal age distribution")
age_dist_survivors = train_df[train_df["Survived"] == 1]["Age"].dropna()

age_dist_deads = train_df[train_df["Survived"] == 0]["Age"].dropna()



ax = sns.distplot(age_dist_deads, kde=False, color='k', label="Deads")

ax = sns.distplot(age_dist_survivors, kde=False, color='g', label="Survivors")

ax.legend()

plt.title("Age distribution")
number_female = train_df[train_df["Sex"] == "female"].size

female_survivors_cnt = train_df[(train_df["Sex"] == "female") & (train_df["Survived"] == 1)].size



number_male = train_df[train_df["Sex"] == "male"].size

male_survivors_cnt = train_df[(train_df["Sex"] == "male") & (train_df["Survived"] == 1)].size



print("There was {} females on board, of which {} survived ({:.2f}%)".format(number_female, female_survivors_cnt, female_survivors_cnt/number_female*100))

print("There was {} males on board, of which {} survived ({:.2f}%)".format(number_male, male_survivors_cnt, male_survivors_cnt/number_male*100))
ax = sns.countplot(data=train_df, x="Survived", hue="Sex")

plt.title("Guess who is screwed again?")
# Fill missing values

train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)

test_df["Age"].fillna(test_df["Age"].mean(), inplace=True)
def preprocess_logistic(df, test=False, nonlinear=False):

    X_logistic = df.copy()

    

    # Remove useless features

    X_logistic.drop("PassengerId", axis=1, inplace=True)

    X_logistic.drop("Name", axis=1, inplace=True)



    # Map categorical values to number values (Logistic Regression can only deal with numbers)

    X_logistic["Sex"] = X_logistic["Sex"].map({

        "female": 0,

        "male": 1

    })



    X_logistic["Embarked"] = X_logistic["Embarked"].map({

        "S": 0,

        "C": 1,

        "Q": 2

    })



    cabin_map = {}

    i = 0

    for cabin_value in X_logistic["Cabin"].unique():

        cabin_map[cabin_value] = i

        i += 1

    X_logistic["Cabin"] = X_logistic["Cabin"].map(cabin_map)

    

    # Handle the last NaN values

    if not test:

        X_logistic.dropna(inplace=True)

    else:

        X_logistic.fillna(0, inplace=True)

    

    try:

        y = X_logistic["Survived"]

        X_logistic.drop("Survived", axis=1, inplace=True)

    except (KeyError, ValueError):

        # Probably a test dataset

        y = None

    

    if nonlinear:

        # Inject some non-linearity by adding some polynomial terms

        X_logistic["SexAndAge"] = X_logistic["Age"] * X_logistic["Sex"]

        X_logistic["SexAndPclass"] = X_logistic["Pclass"] * X_logistic["Sex"]

        

    scaler = StandardScaler()

    X = pd.DataFrame(scaler.fit_transform(X_logistic), columns=X_logistic.axes[1])

    

    return (X, y)
#X_train, X_test, y_train, y_test = train_test_split()
X_train, y_train = preprocess_logistic(train_df, nonlinear=True)



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.05)



X_subm, y_subm = preprocess_logistic(test_df, test=True)
nostradamus = LogisticRegression()

nostradamus.fit(X_train, y_train)
prophecy = nostradamus.predict(X_test)

print("Training score: {}".format(nostradamus.score(X_train, y_train)))

print("Validation score: {} (testing on {} samples)".format(nostradamus.score(X_test, y_test), len(X_test)))
prophecy
nostra_coefs = pd.DataFrame(nostradamus.coef_, columns=X_train.columns)

nostra_coefs.ix[0].plot(kind='bar')