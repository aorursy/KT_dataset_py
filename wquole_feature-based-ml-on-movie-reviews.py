import pickle

from os import path

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
filepath = "../input/sklearn-data.pickle"



data = pickle.load(open(path.abspath(filepath), "rb"))

X_train, y_train, X_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
# converting a collection of text docs to tokenized matrices

vector = HashingVectorizer(stop_words="english", binary=True)

X_train = vector.fit_transform(X_train)

X_test = vector.fit_transform(X_test)
# initialize DTC classifier  with entropy for the information gain (default='gini' for Gini impurity)

dtc_classifier = DecisionTreeClassifier(criterion="entropy")



# fit the transformed data

dtc_classifier.fit(X_train,y_train)



# make predictions for the test set

dtc_y_pred = dtc_classifier.predict(X_test)



print("The accuracy for this classifier:\t{}".format(accuracy_score(dtc_y_pred, y_test)))
bnb_classifier = BernoulliNB()



# fit the transformed data

bnb_classifier.fit(X_train,y_train)



# make predictions for the test set

bnb_y_pred = bnb_classifier.predict(X_test)



# checka accuracy

print("The accuracy for this classifier:\t{}".format(accuracy_score(bnb_y_pred, y_test)))