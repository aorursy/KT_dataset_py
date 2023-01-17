import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pandas.plotting import parallel_coordinates

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression



plt.style.use('ggplot') # make plots look better
df = pd.read_csv("../input/iris/Iris.csv")
df.head()
df.describe()
df.info()
# class distribution

print(df.groupby('Species').size())
# Train-Test Split

train, test = train_test_split(df, test_size = 0.4, stratify = df['Species'], random_state = 42)
# Hist

fig, axs = plt.subplots(2, 2)

axs[0,0].hist(train['SepalLengthCm'], bins = 10)

axs[0,0].set_title('Sepal Length')

axs[0,1].hist(train['SepalWidthCm'], bins = 10)

axs[0,1].set_title('Sepal Width')

axs[1,0].hist(train['PetalLengthCm'], bins = 10)

axs[1,0].set_title('Petal Length')

axs[1,1].hist(train['PetalWidthCm'], bins = 10)

axs[1,1].set_title('Petal Width')

# add some spacing between subplots

fig.tight_layout(pad=1.0)
# Box Plot

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

sns.boxplot(x = 'Species', y = 'SepalLengthCm', data = train, order = classes, ax = axs[0,0], palette="Set1")

sns.boxplot(x = 'Species', y = 'SepalWidthCm', data = train, order = classes, ax = axs[0,1], palette="Set1")

sns.boxplot(x = 'Species', y = 'PetalLengthCm', data = train, order = classes, ax = axs[1,0], palette="Set1")

sns.boxplot(x = 'Species', y = 'PetalWidthCm', data = train,  order = classes, ax = axs[1,1], palette="Set1")

# add some spacing between subplots

fig.tight_layout(pad=6.0)

# Violin Plot SepalLengthCm

# Violin Plot

sns.violinplot(x="Species", y="SepalLengthCm", data=train, size=5, order = classes, palette="Set1")
sns.violinplot(x="Species", y="PetalLengthCm", data=train, size=5, order = classes, palette="Set1")
sns.violinplot(x="Species", y="PetalWidthCm", data=train, size=5, order = classes, palette="Set1")
# Scatter plot

sns.pairplot(train.iloc[1:, 1:], hue="Species", height = 2, palette="Set1");
# parallel coordinate plot

parallel_coordinates(train.iloc[1:, 1:], "Species", color = ['red','blue', 'green']);
# Correlation Matrix

corrmat = train.corr()

sns.heatmap(corrmat.iloc[1:, 1:], annot = True, square = True, )

plt.xticks(rotation=45) 



plt.show()
# Separate class label and features

X_train = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_train = train.Species

X_test = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_test = test.Species
# Classification Tree

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)

mod_dt.fit(X_train,y_train)

prediction=mod_dt.predict(X_test)

print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))
plt.figure(figsize = (10,8))

plot_tree(mod_dt, feature_names = features, class_names = classes, filled = True);
#Confusion Matrix

# https://towardsdatascience.com/exploring-classifiers-with-python-scikit-learn-iris-dataset-2bcb490d2e1b

#Through this matrix, we see that there is one versicolor which we predict to be virginica.

disp = metrics.plot_confusion_matrix(mod_dt, X_test, y_test,

                                 display_labels=classes,

                                 cmap=plt.cm.Blues,

                                 normalize=None)

disp.ax_.set_title('Decision Tree Confusion matrix, without normalization')
sns.FacetGrid(df,  

    hue="Species", palette="Set1").map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()



plt.show()
sns.FacetGrid(df,  

    hue="Species", palette="Set1").map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()



plt.show()
labels = np.asarray(df.Species)
# use LabelEncoder to label "String" Species as Numerical ones.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(labels)



# apply encoding to labels

labels = le.transform(labels)
df.sample(5)
labels
df_selected = df.drop(['SepalLengthCm', 'SepalWidthCm', "Id", "Species"], axis=1)
df_features = df_selected.to_dict(orient='records')
df_features[:5]
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

features = vec.fit_transform(df_features).toarray()
from sklearn.model_selection import train_test_split



features_train, features_test, labels_train, labels_test = train_test_split( features, labels, test_size=0.33, random_state=42)
# Now Classify again

from sklearn.ensemble import RandomForestClassifier



# initialize

clf = RandomForestClassifier()



# train the classifier using the training data

clf.fit(features_train, labels_train)
# compute accuracy using test data

acc_test = clf.score(features_test, labels_test)



print ("Test Accuracy:", acc_test)
# compute accuracy using training data

acc_train = clf.score(features_train, labels_train)



print ("Train Accuracy:", acc_train)
flower = [[5.2,0.9]]

class_code = clf.predict(flower)
class_code
decoded_class = le.inverse_transform(class_code)

print (decoded_class) # ['Iris-versicolor']
pred = clf.predict(features_test)
#Evaluation

from sklearn.metrics import recall_score, precision_score



precision = precision_score(labels_test, pred, average="weighted")

recall = recall_score(labels_test, pred, average="weighted")



print ("Precision:", precision) # Precision: 0.98125

print ("Recall:", recall) # Recall: 0.98
clf = RandomForestClassifier(

    min_samples_split=4,

    criterion="entropy"

)

clf.fit(features_train, labels_train)
acc_test = clf.score(features_test, labels_test)

acc_train = clf.score(features_train, labels_train)

print ("Train Accuracy:", acc_train)

print ("Test Accuracy:", acc_test)
pred = clf.predict(features_test)
precision = precision_score(labels_test, pred, average="weighted")

recall = recall_score(labels_test, pred, average="weighted")



print ("Precision:", precision)

print ("Recall:", recall)
# Now SVC

from sklearn.svm import SVC

clf = SVC()
clf.fit(features_train, labels_train)



# find the accuracy of the model

acc_test = clf.score(features_test, labels_test)

acc_train = clf.score(features_train, labels_train)

print ("Train Accuracy:", acc_train)

print ("Test Accuracy:", acc_test)
# compute predictions on test features

pred = clf.predict(features_test)



# predict our new unique iris flower

flower = [[5.2,0.9]]

class_code = clf.predict(flower)

class_code
from sklearn.metrics import recall_score, precision_score



precision = precision_score(labels_test, pred, average="weighted")

recall = recall_score(labels_test, pred, average="weighted")



print ("Precision:", precision)

print ("Recall:", recall)