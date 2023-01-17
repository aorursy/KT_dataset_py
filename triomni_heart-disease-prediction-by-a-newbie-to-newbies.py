#@Rita
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/heart.csv')
df.head()
# We must clean the NaN values, since we cannot train

# our model with unknown values.

# Luckily, there is nothing to clean.

df.info() , df.isna().sum()
# We have no need to encode labels since 

# the categories are fine (integers).
# GOAL: presence of heart disease in a patient given the explanatory variables

# which are age, sex, and so on.

# Therefore, target is YES or NO (1/0) depending whether a patient has got the disease or not.

df.target.value_counts()
# The best way to explore data is via plots

# so we gonna plot some stuff
# First of all let's see how many zeros and ones do we have...

negative_target = len(df[df.target == 0])

positive_target = len(df[df.target == 1])

print("Percentage of Patients that do not have Heart Disease: {:.3f}%".format((negative_target / (len(df.target))*100)))

print("Percentage of Patients that have Heart Disease: {:.3f}%".format((positive_target / (len(df.target))*100)))

sns.countplot(x = "target", data = df, palette = "pastel")

plt.xlabel("Target (0 = no, 1= yes)")

plt.ylabel("count")

plt.show()
# Males vs Females

female_patient = len(df[df.sex == 0])

male_patient = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.3f}%".format((female_patient / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.3f}%".format((male_patient / (len(df.sex))*100)))

sns.countplot(x = 'sex', data = df, palette = "pastel")

plt.xlabel("Sex (0 = female, 1= male)")

plt.ylabel("count")

plt.show()
for x in range (0, 4):

    print("% of patients that show positivity with chest pain {}: {:.3f}%".format(x, 100*(((df['cp'] == x) & df['target'] == 1).sum())/((df['target'] == 1).sum())))
for x in range (0, 4):

    print("% of patients that show negativity with chest pain {}: {:.3f}%".format(x, 100*(((df['cp'] == x) & (df['target'] == 0)).sum())/((df['target'] == 0).sum())))
print("% of patients that show positivity with trestbps > 130: {:.3f}%".format

      (100*(((df['trestbps'] > 130) & df['target'] == 1).sum())

       /((df['target'] == 1).sum())))

print("% of patients that show positivity with trestbps <= 130: {:.3f}%".format

      (100*(((df['trestbps'] <= 130) & df['target'] == 1).sum())

       /((df['target'] == 1).sum())))
print("% of patients that show positivity with chol > 200: {:.3f}%".format

      (100*(((df['chol'] > 200) & df['target'] == 1).sum())

       /((df['target'] == 1).sum())))

print("% of patients that show positivity with chol <= 200: {:.3f}%".format

      (100*(((df['chol'] <= 200) & df['target'] == 1).sum())

       /((df['target'] == 1).sum())))
for x in range (0, 2):

    print("% of patients that show positivity with fbs {}: {:.3f}%".format(x, 100*(((df['fbs'] == x) & df['target'] == 1).sum())/((df['target'] == 1).sum())))
for x in range (0, 3):

    print("% of patients that show positivity with restecg {}: {:.3f}%".format(x, 100*(((df['restecg'] == x) & df['target'] == 1).sum())/((df['target'] == 1).sum())))
print("% of patients that show positivity with thalach > 100: {:.3f}%".format

      (100*(((df['thalach'] > 100) & df['target'] == 1).sum())

       /((df['target'] == 1).sum())))

print("% of patients that show positivity with thalach <= 100: {:.3f}%".format

      (100*(((df['thalach'] <= 100) & df['target'] == 1).sum())

       /((df['target'] == 1).sum())))
for x in range (0, 2):

    print("% of patients that show positivity with exang {}: {:.3f}%".format(x, 100*(((df['exang'] == x) & df['target'] == 1).sum())/((df['target'] == 1).sum())))
for x in range (0, 3):

    print("% of patients that show positivity with slope {}: {:.3f}%".format(x, 100*(((df['slope'] == x) & df['target'] == 1).sum())/((df['target'] == 1).sum())))

print("We can suppose that Slope = 2 is likely to sign Heart disease.")   
for x in range (0, 3):

    print("% of patients that show positivity with ca {}: {:.3f}%".format(x, 100*(((df['ca'] == x) & df['target'] == 1).sum())/((df['target'] == 1).sum())))
# We have got more positive samples than negative,

# And we have more females than males in the dataset...

# but what is the proportion of positivity by sex?
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(20,6), color=['#99ccff','#ffcc99'])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex')

plt.ylabel('Frequency')

plt.show()
# Intuitively, age must be an important factor:

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.countplot(x = 'age', data = df, palette = "pastel")

plt.xlabel("Age")

plt.ylabel("count")

plt.show()
pd.crosstab(df.age, df.target).plot(kind = "bar", figsize = (20,6), color=['#99ccff','#ffcc99'])

plt.title('Heart Disease Frequency by Age')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
# Now, let's dive into more medical categories: 

# What type of chest pain is likely to show a positive result?
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(20,6), color=['#99ccff','#ffcc99'])

plt.title('Heart Disease Frequency for Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.ylabel('Frequency')

plt.show()
# Chest pain by age?

pd.crosstab(df.age, df.cp).plot(kind = "bar", figsize = (20,6), color=['#99ccff','#ffcc99', '#ffccff','#99ff99'])

plt.title('Chest Pain type by Age')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
pd.crosstab(df.ca,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for # of major vessels')

plt.xlabel('Number of major vessels')

plt.ylabel('Frequency')

plt.show()
# The plot below can be used to determine the correlation between the different features of the dataset. 

# From the above set we can also find out the features which have the most and the least effect on the target 

# feature (whether the patient have heart diseases or not).
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)

_ = plt.title('Correlation')
# target is highly correlated with CP, THALACH, EXANG, OLDPEAK.

# Secondly with CA, SLOPE, THAL.
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
# It is a classification problem, and more precisely a BINARY CLASSIFICATION.

# we divide into two subsets: the training data and the testing data

# X are the explanatory variables, Y is the response variable ('target'):

X = df.drop('target', axis=1)  # everything except target.

Y = df['target']               # only target.





# Since the amount of data is not extremely large, we will use a small test_size (0.10-0.15).

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15) 
# Firstly, let's take Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train, Y_train)

rfc.score(X_test, Y_test)
rfc_predictions = rfc.predict(X_test)

test_error = accuracy_score(rfc_predictions, Y_test)

# test_error == rfc.score(X_test, Y_test)

test_error
# Cross validation: see how it works in average:

cross_validation_scores = cross_val_score(rfc, X, Y, cv=10)

print('Average accuracy for Random Forest: %0.4f' %cross_validation_scores.mean())

print(cross_validation_scores)
# no overfitting!

# since test_error is lower than test_train (= cross_validation_scores.mean()).

# Let's try a smaller test_size:

rfc_2 = RandomForestClassifier(n_estimators=1000)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05) 

rfc_2.fit(X_train, Y_train)

rfc_2.score(X_test, Y_test), cross_val_score(rfc_2, X, Y, cv=10).mean()

# overfitting.
# Tuning of random forest classifier:

# 1. n_estimators HIGH.  The more estimators you give it, the better it will do.

# 2. max_features.  It may have a large impact on the behavior of the RF because it decides 

# how many features each tree in the RF considers at each split. Default is 'sqrt'.

tuned_rfc = RandomForestClassifier(n_estimators = 5000, oob_score = True,

                                   max_depth = None, max_features = 'sqrt')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05) 

tuned_rfc.fit(X_train, Y_train)

tuned_rfc.score(X_test, Y_test), cross_val_score(tuned_rfc, X, Y, cv=10).mean()



# I have tried max_features = 'log2' and it does not improve, so the best we can do in this

# case is increasing n_estimators.
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

for i in range (1,10):

    knn = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn.fit(X_train, Y_train)

    prediction = knn.predict(X_train)



    print("{} NN Score: {:.4f}%".format(i, knn.score(X_test, Y_test)*100))
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(X_train, Y_train)



print("Test Accuracy of SVM Algorithm: {:.4f}%".format(svm.score(X_test, Y_test)*100))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, Y_train)

print("Accuracy of Naive Bayes: {:.4f}%".format(nb.score(X_train, Y_train)*100))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, Y_train)

print("Decision Tree Test Accuracy {:.4f}%".format(dtc.score(X_train, Y_train)*100))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(X_train, Y_train)

print("Random Forest Algorithm Accuracy Score : {:.4f}%".format(rf.score(X_train, Y_train)*100))
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_validate

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True,gamma='scale'),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=100),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression(solver='lbfgs')]



scores = []

for clf in classifiers:

    clf.fit(X_train, Y_train)

    cv_results = cross_validate(clf, X_test, Y_test, cv=5, return_train_score=True)

    scores.append(np.mean(cv_results['test_score']))

    

sns.barplot(y=[n.__class__.__name__  for n in classifiers], x=scores, orient='h')
scores
# We observe that logistics regression is the best BUT! 🙈 

# If we use a LINEAR kernel in SVC it should work better!

svm = SVC(probability=True, kernel='linear',gamma='scale')

svm.fit(X_train, Y_train)



print("Test Accuracy of SVM Algorithm: {:.4f}%".format(svm.score(X_test, Y_test)*100))

cv_result = cross_val_score(svm, X, Y, cv=10)

print(cv_result.mean())
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

X = df.drop('target', axis=1) # everything except target.

Y = df['target']               # only target.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

nb.fit(X_train, Y_train)



print("Accuracy of Naive Bayes: {:.4f}%".format(nb.score(X_train, Y_train)*100))
# Which SVC kernel is better?

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

svm_linear = SVC(probability=True, kernel='linear',gamma='scale')

svm_linear.fit(X_train, Y_train)



print("Test Accuracy of SVM Algorithm: {:.4f}%".format(svm_linear.score(X_test, Y_test)*100))

cv_result_linear = cross_val_score(svm_linear, X, Y, cv=10)
svm_poly = SVC(probability=True, kernel='poly',degree=3,gamma='scale')

svm_poly.fit(X_train, Y_train)



print("Test Accuracy of SVM Algorithm: {:.4f}%".format(svm_poly.score(X_test, Y_test)*100))

cv_result_poly = cross_val_score(svm_poly, X, Y, cv=3)
svm_rbf = SVC(probability=True, kernel='rbf',gamma='scale')

svm_rbf.fit(X_train, Y_train)



print("Test Accuracy of SVM Algorithm: {:.4f}%".format(svm_rbf.score(X_test, Y_test)*100))

cv_result_rbf = cross_val_score(svm_rbf, X, Y, cv=10)
# order by

print(cv_result_linear.mean(), cv_result_poly.mean(), cv_result_rbf.mean())
X_test.head()
Y_test.head()
print(nb.predict((X_test.loc[[227]]).values.tolist()),

nb.predict((X_test.loc[[269]]).values.tolist()),

nb.predict((X_test.loc[[262]]).values.tolist()),

nb.predict((X_test.loc[[300]]).values.tolist()),

nb.predict((X_test.loc[[192]]).values.tolist()))
# 😮
pairs = sns.pairplot(df)

pairs
pairs.savefig('a')

# age with chol.

# trestbps with chol.

# thalach with chol.
from sklearn.decomposition import PCA

pca = PCA()
df_pca = pca.fit_transform(df)

y_variance = pca.explained_variance_ratio_

pd.DataFrame(pca.components_, columns=df.columns)
# See that 0 has 0.99 of CHOL

# 1 has -0.97 of THALACH

# 2 has 0.98 of TRESTBPS

# recall:

sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)

# Possibly others had high values due to multicollinearity.
sns.barplot(x=[i for i in range(len(y_variance))], y=y_variance)

plt.title("PCA")
X = df.drop('target', axis=1) 

pca = PCA()

pca.fit(X)

df_pca = pca.transform(X)

print("original shape:   ", X.shape)

print("transformed shape:", df_pca.shape)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');



# we can observe the cumulative explained variance.
np.cumsum(pca.explained_variance_ratio_)[2], np.cumsum(pca.explained_variance_ratio_)[3]

# With 3 components it's more than enough, even with 2 we can explain the 98% of the

# variability in the data! 😮
sns.barplot(np.arange(0,14),pca.explained_variance_ratio_)
from sklearn.datasets import make_blobs

from sklearn import decomposition
# I decided to take 2 components:

pca = decomposition.PCA(n_components=2)

X_pca = pca.fit_transform(X)



df_pca = pd.DataFrame(data = X_pca , 

        columns = ['PC1', 'PC2'])

df_pca['Cluster'] = Y

df_pca.head()
pca.explained_variance_ratio_
pc_df = pd.DataFrame({'var': pca.explained_variance_ratio_,

             'PC':['PC1','PC2']})

sns.barplot(x='PC',y="var", 

           data=pc_df, color="c");
sns.lmplot( x="PC1", y="PC2",

  data=df_pca, 

  fit_reg=False, 

  hue='Cluster', # color by cluster

  legend=True,

  scatter_kws={"s": 80}) # specify the point size
from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel



lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)

model = SelectFromModel(lsvc, prefit=True)

feature_index = model.get_support()

feature_index # what features have an explicit influence
# We take them 

X.head()

X_up = X[X.columns[feature_index]]

X[X.columns[feature_index]].head()
# First, let's see if our tuned RFC improves...

tuned_rfc = RandomForestClassifier(n_estimators = 1000, oob_score = True,

                                   max_depth = None, max_features = 'sqrt')

X_train, X_test, Y_train, Y_test = train_test_split(X_up, Y, test_size=0.05) 

tuned_rfc.fit(X_train, Y_train)

tuned_rfc.score(X_test, Y_test), cross_val_score(tuned_rfc, X_up, Y, cv=10).mean()
# Seems like not 😢
# Let's try our best option SVC: 

X_train, X_test, Y_train, Y_test = train_test_split(X_up, Y, test_size=0.05) 



svm = SVC(probability=True, kernel='linear',gamma='scale')

svm.fit(X_train, Y_train)



print("Test Accuracy of SVM Algorithm: {:.4f}%".format(svm.score(X_test, Y_test)*100))

cv_result = cross_val_score(svm, X, Y, cv=10)

print(cv_result.mean())
# Not better 😑 OK.
# First, let's see an example with 3 clusters:

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

kmeans.fit(X)
kmeans.labels_
kmeans.inertia_ # that's the cost, the lower - the better.
df['KMEANS'] = kmeans.labels_
df.head()
plt.scatter(x=df_pca['PC1'], y=df_pca['PC2'], c=kmeans.labels_*2 )
from sklearn import metrics

from sklearn.metrics import pairwise_distances

kmeans_cost = np.array([])

calinskis = np.array([])

for k in range(2,20):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(X)

    kmeans_cost = np.append(kmeans_cost, kmeans.inertia_)

    calinskis = np.append(calinskis, metrics.calinski_harabaz_score(pc, kmeans.labels_))

plt.scatter(x=df_pca['PC1'], y=df_pca['PC2'], c=kmeans.labels_*2 )
x = np.arange(2,20)

y = kmeans_cost

sns.barplot(x,y)
plt.scatter(x, calinskis), calinskis
# We take the best k: where the cost does not vary much. See it with calinski

# where do we have the lowest value? 11 groups:

kmeans = KMeans(n_clusters=11)

kmeans.fit(X)

kmeans_cost = np.append(kmeans_cost, kmeans.inertia_)

plt.scatter(x=df_pca['PC1'], y=df_pca['PC2'], c=kmeans.labels_*2 )
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import to_categorical

import keras
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)



accuracies = []

losses = []

for i in range(0, 5):

    model = Sequential()

    model.add(Dense(5, input_dim=X_train.shape[1], activation='relu'))    

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

    model.fit(X_train, Y_train, epochs=400, batch_size=32, verbose=1)



    loss, acc = model.evaluate(X_test, Y_test, batch_size=32)



accuracies = []

losses = []

# It is better for internal nodes to have linear and rectified activations.

# In binary classification, the last level has to have just ONE node.

# And it's better to have there a sigmoid-like activation to distinguish between the

# 2 states (0 and 1).

internal = ['relu', 'tanh', 'linear']

last = ['sigmoid', 'tanh', 'softsign']

optimizer = ['RMSprop', 'SGD', 'adam']



for internal_activation_func in range(len(internal)):

    for last_activation_func in range(len(last)):

        for j in range(0,5):

            print("INTERNAL: " + internal[internal_activation_func])

            print("LAST: " + last[last_activation_func])

            

            model = Sequential()

            model.add(Dense(5, input_dim=X_train.shape[1], activation=internal[internal_activation_func]))    

            model.add(Dense(1, activation=last[last_activation_func]))

            model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    



            # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

            model.fit(X_train, Y_train, epochs=400, batch_size=32, verbose=1)



            loss, acc = model.evaluate(X_test, Y_test, batch_size=32)



            accuracies.append(acc)

            losses.append(loss)



# After 20 minutes... 
print("NEURAL NETWORK 1:")

print("\tINTERNAL: " + internal[0])

print("\tLAST: " + last[0])

print("\tAccuracy: " + str(np.mean(accuracies[0:5])))



print("NEURAL NETWORK 2:")

print("\tINTERNAL: " + internal[0])

print("\tLAST: " + last[1])

print("\tAccuracy: " + str(np.mean(accuracies[5:10])))



print("NEURAL NETWORK 3:")

print("\tINTERNAL: " + internal[0])

print("\tLAST: " + last[2])

print("\tAccuracy: " + str(np.mean(accuracies[10:15])))
print("NEURAL NETWORK 4:")

print("\tINTERNAL: " + internal[1])

print("\tLAST: " + last[0])

print("\tAccuracy: " + str(np.mean(accuracies[15:20])))



print("NEURAL NETWORK 5:")

print("\tINTERNAL: " + internal[1])

print("\tLAST: " + last[1])

print("\tAccuracy: " + str(np.mean(accuracies[20:25])))



print("NEURAL NETWORK 6:")

print("\tINTERNAL: " + internal[1])

print("\tLAST: " + last[2])

print("\tAccuracy: " + str(np.mean(accuracies[25:30])))
print("NEURAL NETWORK 7:")

print("\tINTERNAL: " + internal[2])

print("\tLAST: " + last[0])

print("\tAccuracy: " + str(np.mean(accuracies[20:25])))



print("NEURAL NETWORK 8:")

print("\tINTERNAL: " + internal[2])

print("\tLAST: " + last[1])

print("\tAccuracy: " + str(np.mean(accuracies[25:30])))



print("NEURAL NETWORK 9:")

print("\tINTERNAL: " + internal[2])

print("\tLAST: " + last[2])

print("\tAccuracy: " + str(np.mean(accuracies[35:40])))
# Horrible neural networks, bye 🤯
# Let's do a single neural network with several layers and with act functions

# which have given the best results in the previous attempt, which are TANH and SIGMOID:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

accuracies = []

losses = []

for j in range(0,5):

    model = Sequential()

    model.add(Dense(5, input_dim=X_train.shape[1], activation='tanh'))  

    model.add(Dense(5, activation='tanh'))    

    model.add(Dense(5, activation='tanh'))    

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    



    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

    model.fit(X_train, Y_train, epochs=400, batch_size=32, verbose=1)



    loss, acc = model.evaluate(X_test, Y_test, batch_size=32)



    accuracies.append(acc)

    losses.append(loss)
np.mean(accuracies) # well...
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

accuracies = []

losses = []

for j in range(0,5):

    model = Sequential()

    model.add(Dense(10, input_dim=X_train.shape[1], activation='tanh'))  

    model.add(Dense(2, activation='sigmoid'))     

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    



    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

    model.fit(X_train, Y_train, epochs=400, batch_size=32, verbose=1)



    loss, acc = model.evaluate(X_test, Y_test, batch_size=32)



    accuracies.append(acc)

    losses.append(loss)
np.mean(accuracies) 
model.summary()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

accuracies = []

losses = []

for j in range(0,5):

    model = Sequential()

    model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))  

    model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    



    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

    model.fit(X_train, Y_train, epochs=400, batch_size=32, verbose=1)



    loss, acc = model.evaluate(X_test, Y_test, batch_size=32)



    accuracies.append(acc)

    losses.append(loss)
# approx 0.5x
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

accuracies = []

losses = []

for j in range(0,5):

    model = Sequential()

    model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))  

    model.compile(loss= 'binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    



    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

    model.fit(X_train, Y_train, epochs=400, batch_size=32, verbose=1)



    loss, acc = model.evaluate(X_test, Y_test, batch_size=32)



    accuracies.append(acc)

    losses.append(loss)
np.mean(accuracies) #pff
# that's all