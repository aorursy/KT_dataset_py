import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.feature_extraction.text import CountVectorizer
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Getting output variable

y = train['Survived']

train = train.drop(['Survived'],axis=1)
# Concatenating data sets

data = pd.concat([train, test], axis=0)
vectorizer = CountVectorizer(min_df=1,lowercase=True,max_features=150)

Names = vectorizer.fit_transform(data["Name"])

data = pd.merge(data,pd.DataFrame(Names.toarray()),left_index=True, right_index=True)



# Drop names because we have synthetized them

data = data.drop(["Name"],axis=1)
# Generate features from categorical features

data = pd.get_dummies(data)



# Replace NAs with means

data = data.fillna(data.mean())



# Sort dataframe on PassengerId

data = data.sort("PassengerId",ascending=True)
# Re-split dataset to X_train and X_test

X_train = data[:train.shape[0]]

X_test = data[train.shape[0]:]
# Instanciate all classifiers 

rfc = RandomForestClassifier(n_jobs=-1,n_estimators=1000)

clf3 = GaussianNB()

clf1 = LogisticRegression(random_state=1)

gbc = GradientBoostingClassifier()

abc = AdaBoostClassifier()



# Instanciate voting classifier with classifiers 

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rfc', rfc), ('gnb', clf3),('gbc', gbc), ('abc', abc)], voting='soft')

eclf1 = eclf1.fit(X_train, y)
# Fit training data and print score 

eclf1.fit(X_train,y)

#print eclf1.score(X_train,y)
# Let's predict ouputs on test set

predictions = eclf1.predict(X_test)

submission = pd.DataFrame({"PassengerId": X_test["PassengerId"],"Survived":predictions})

submission = submission.set_index("PassengerId")



submission.to_csv('submission.csv')