"""

by yichu

"""

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



#matplotlib.pyplot.ion()



pre_path = 'D:/code-study/kaggle/titanic/'

pre_path = '../input/'

cols = ['Sex', 'Pclass', 'Cabin_known', 'Large_Family', 'Parch',

        'SibSp', 'Young', 'Alone', 'Shared_ticket', 'Child']

has_lable_cols = np.append(['Survived'], cols)





def load_data():

    train = pd.read_csv(pre_path + "train.csv")

    print(train.info())

    test = pd.read_csv(pre_path + "test.csv")

    # modify combine's data won't influence train or test's data, and vice versa

    combine = pd.concat([train, test])

    return train, test, combine





def fill_missing_values(train, test):

    median_value = combine['Fare'][combine['Pclass'] == 3].dropna().median()

    combine['Fare'] = combine['Fare'].fillna(value=median_value)





def generate_derived_features(combine,len_train):

    combine['Child'] = combine['Age'] <= 10

    combine['Cabin_known'] = combine['Cabin'].isnull() == False

    combine['Age_known'] = combine['Age'].isnull() == False

    combine['Family'] = combine['SibSp'] + combine['Parch']

    combine['Alone'] = (combine['SibSp'] + combine['Parch']) == 0

    combine['Large_Family'] = (combine['SibSp'] > 2) | (combine['Parch'] > 3)



    combine['Deck'] = combine['Cabin'].str[0]

    combine['Deck'] = combine['Deck'].fillna(value='U')

    combine["Deck"] = combine["Deck"].astype("category")

    combine["Deck"].cat.categories = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    combine["Deck"] = combine["Deck"].astype("int")



    combine['Ttype'] = combine['Ticket'].str[0]

    combine['Title'] = combine['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]





    combine['Fare_cat'] = pd.DataFrame(np.floor(np.log10(combine['Fare'] + 1))).astype('int')



    combine['Bad_ticket'] = combine['Ttype'].isin(['3', '4', '5', '6', '7', '8', 'A', 'L', 'W'])

    combine['Young'] = (combine['Age'] <= 30) | (combine['Title'].isin(['Master', 'Miss', 'Mlle']))

    combine['Shared_ticket'] = np.where(combine.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)

    combine['Ticket_group'] = combine.groupby('Ticket')['Name'].transform('count')

    combine['Fare_eff'] = combine['Fare'] / combine['Ticket_group']

    combine['Fare_eff_cat'] = np.where(combine['Fare_eff'] > 16.0, 2, 1)

    combine['Fare_eff_cat'] = np.where(combine['Fare_eff'] < 8.5, 0, combine['Fare_eff_cat'])



    combine["Sex"] = combine["Sex"].astype("category")

    combine["Sex"].cat.categories = [0, 1]

    combine["Sex"] = combine["Sex"].astype("int")



    combine["Embarked"] = combine["Embarked"].astype("category")

    combine["Embarked"].cat.categories = [0, 1, 2]

    combine["Embarked"] = combine["Embarked"].astype("int")

    train = combine.iloc[:len_train]

    test = combine.iloc[len_train:]

    return train,test





def train_model(train):

    training, testing = train_test_split(train, test_size=0.2, random_state=0)



    df = training.loc[:, has_lable_cols].dropna()

    X = df.loc[:, cols]

    y = np.ravel(df.loc[:, ['Survived']])



    classifier_LR = LogisticRegression()

    classifier_LR = classifier_LR.fit(X, y)

    score_LR = cross_val_score(classifier_LR, X, y, cv=5).mean()

    print('------------------------')

    print(score_LR)

    return classifier_LR





def predict(classifier_LR, test):

    df2 = test.loc[:, cols].fillna(method='pad')

    surv_pred = classifier_LR.predict(df2)

    surv_pred=surv_pred.astype('int')



    submit = pd.DataFrame({'PassengerId': test.loc[:, 'PassengerId'],

                           'Survived': surv_pred.T})

    submit.info()

    #submit.to_csv(pre_path + "submit.csv", index=False)

# entry

train, test, combine = load_data()
fill_missing_values(train, test)

train, test=generate_derived_features(combine, len(train))



classifier_LR=train_model(train)
predict(classifier_LR, test)