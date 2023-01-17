#this dataset from http://archive.ics.uci.edu/ml/datasets/Bank+Marketing

bank_additional = 'bank-additional-full.csv'
#import from python library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns
#Create dataframe from the data set

df = pd.read_csv ('../input/bank-additional-full.csv', sep=';', decimal ='.', header =0, names = 

                 ['age', 'job', 'marital', 'education', 'default','housing','loan','contact','month',

                 'day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx',

                 'cons.conf.idx','euribor3m','nr.employed','target'])
#look for data types

print (df.dtypes)
#check the import process already correct

print (df.head())
#check how many sample and attribute in the data

print (df.shape)
#statistic description for numerical variables

print (df.describe())
#The data is already clean from missing values

print (df.isnull().sum())
#Further check for sanity check, typo errors, redundant white space



print (df ['age'].value_counts())

print (df ['job'].value_counts())

print (df ['marital'].value_counts())

print (df ['education'].value_counts())

print (df ['default'].value_counts())

print (df ['housing'].value_counts())

print (df ['loan'].value_counts())

print (df ['contact'].value_counts())

print (df ['month'].value_counts())

print (df ['day_of_week'].value_counts())

print (df ['duration'].value_counts())

print (df ['campaign'].value_counts())

print (df ['pdays'].value_counts())

print (df ['previous'].value_counts())

print (df ['poutcome'].value_counts())

print (df ['emp.var.rate'].value_counts())

print (df ['cons.price.idx'].value_counts())

print (df ['cons.conf.idx'].value_counts())

print (df ['euribor3m'].value_counts() )

print (df ['nr.employed'] .value_counts())

print (df ['target'].value_counts())
#Checking data distribution for categorical value using pie chart

# Data to plot

labels_house = ['yes', 'no', 'unknown']

sizes_house = [2175, 1839, 105]

colors_house = ['#ff6666', '#ffcc99', '#ffb3e6']



labels_loan = ['yes', 'no', 'unknown']

sizes_loan = [665, 3349, 105]

colors_loan = ['#c2c2f0','#ffb3e6', '#66b3ff' ]



labels_contact = ['cellular', 'telephone']

sizes_contact = [2652, 1467]

colors_contact = ['#ff9999','#ffcc99']



labels_default = ['no','unknown','yes']

sizes_default = [3523, 454, 142]

colors_default = ['#99ff99','#66b3ff','#ff6666' ]





# Plot

plt.rcParams.update({'font.size': 15})



plt.figure(0)

plt.pie(sizes_house, labels=labels_house, colors=colors_house, autopct='%1.1f%%', startangle=90, pctdistance=0.8)

plt.title ('Housing Loan')

centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.show()



plt.figure(1)

plt.pie(sizes_loan,labels=labels_loan, colors=colors_loan, autopct='%1.1f%%',startangle=90,pctdistance=0.8)

plt.title ('Personal Loan')

centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.show()



plt.figure(2)

plt.pie(sizes_contact, labels=labels_contact, colors=colors_contact, autopct='%1.1f%%', startangle=90,pctdistance=0.8)

plt.title ('Contact Method')

centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.show()



plt.figure(3)

plt.pie(sizes_default, labels=labels_default, colors=colors_default, autopct='%1.1f%%', startangle=90,pctdistance=0.8)

plt.title ('default')

centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.show()

# exploring loan ownership vs current campaign result/outcome

sns.catplot(x="target", hue="loan", kind="count", data = df)

plt.title ('result by personal loan status')

plt.xlabel("results(y) of current campaign")

plt.show ()
# exploring loan ownership vs current campaign result/outcome

sns.catplot(x="target", hue="housing", kind="count", data = df)

plt.title ('result by housing loan status')# exploring loan ownership vs current campaign result/outcome

plt.show ()

# exploring default status vs current campaign result/outcome

sns.catplot(x="target", hue="default", kind="count", data = df)

plt.title ('result by default status')

plt.xlabel("results(y) of current campaign")

plt.show ()
# clarifying effect of 'yes' value of default status on target as not visible in count plot

ldf = df[(df.default == "yes")]

print (ldf)
#result(y) vs previous campaign outcome(poutcome) vs duration 

g = sns.catplot(x="duration", y="target", row = "poutcome",

                kind="box", orient="h", height=2.5, aspect=5,

                data=df)
# exploring relationship between result and current campaign

sns.catplot(x="target", kind="count", data = df);

plt.title ('results (target count) of current campaign')

plt.xlabel("results(target) of current campaign")



plt.show ()
# exploring job type vs current campaign result/outcome

sns.catplot(x="target", hue="job", kind="count", data = df)

plt.title ('Result by job type of clients')

plt.xlabel("results(y) of current campaign")

plt.show ()
# exploring job type vs current campaign result/outcome

sns.catplot(x="target", hue="education", kind="count", data = df)

plt.title ('Result by education level of clients')

plt.xlabel("results(y) of current campaign")

plt.show ()
# exploring job type vs current campaign result/outcome

fig, ax = plt.subplots()

fig.set_size_inches(25, 8)

sns.countplot(x = 'age', data = df)

ax.set_xlabel('Age', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Age Count Distribution', fontsize=15)

sns.despine()
#from the heatmap we can see the positive relationship between "cons.price.idx-emp.var.rate", "euribor3m-emp.var.rate", "nr.employed-emp.var.rate", "euribor3m-cons.price.idx"



plt.figure(figsize=(11,4))

sns.heatmap(df[["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx","euribor3m", "nr.employed"]].corr(),annot=True)

plt.show()
#scatter matrix for numeric value, we can see that most of the numerical value is scattered



g = sns.pairplot(df[["age", "duration", "campaign", "emp.var.rate", "cons.price.idx", "cons.conf.idx","euribor3m", "nr.employed"]], diag_kind="kde")

plt.show ()
#Creating new dataframe for classification model

dftree = df[['age', 'job', 'marital', 'education', 'default','housing','loan','contact','month',

                 'day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx',

                 'cons.conf.idx','euribor3m','nr.employed','target']]



dftreetarget = dftree['target']

print (dftree.head())

print (dftree.shape)
#In order to pass the data into k-nearest neighbors we need to encode the categorical values to integers

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



#encoding/transforming

dftree['job'] = le.fit_transform(dftree['job'].astype('str'))

dftree['marital'] = le.fit_transform(dftree['marital'].astype('str'))

dftree['education'] = le.fit_transform(dftree['education'].astype('str'))

dftree['default'] = le.fit_transform(dftree['default'].astype('str'))

dftree['housing'] = le.fit_transform(dftree['housing'].astype('str'))

dftree['loan'] = le.fit_transform(dftree['loan'].astype('str'))

dftree['contact'] = le.fit_transform(dftree['contact'].astype('str'))

dftree['month'] = le.fit_transform(dftree['month'].astype('str'))

dftree['day_of_week'] = le.fit_transform(dftree['day_of_week'].astype('str'))

dftree['poutcome'] = le.fit_transform(dftree['poutcome'].astype('str'))

dftree['target'] = le.fit_transform(dftreetarget.astype('str'))



dftreetarget = le.fit_transform(dftreetarget.astype('str')) ## separatly creating a target variable for modelling
print (dftree.head())

print (dftreetarget)
#Before running the classification model we need to drop the target value labels

dftree = dftree.drop('target', axis=1)

print (dftree.head())
#set train and test data 50% train 50% test



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dftree.values, dftreetarget, test_size=0.5,random_state=0)

print (X_train)

print (X_train.shape)

print (y_train)

print (y_train.shape)

print (X_test)

print (y_test)

print (y_test.shape)
#Classification using Decision Tree Classifier parameter tuning

#Using for loop iteration to find best precision for this model

from sklearn.tree import DecisionTreeClassifier



bestprecision = 0



for a in range (2, 5+1):

    for b in range (2, 5+1):

        for c in range (1, 5+1):

            clf = DecisionTreeClassifier(criterion='gini', max_depth=a, min_samples_split=b, min_samples_leaf=c, max_features=None, max_leaf_nodes=None)

            fit = clf.fit(X_train, y_train)

            y_pre = fit.predict(X_test)

            

            from sklearn.metrics import confusion_matrix

            from sklearn.metrics import classification_report

            cm = confusion_matrix(y_test, y_pre)

            print ("max depth = " + str(a) + " min samples split = " + str(b) + "min samples leaf" + str(c))

            print (cm)

            print (classification_report(y_test,y_pre))

            

            precision = round(float(cm.item(3))/float((cm.item(1)+cm.item(3))),2)

            

            if(precision > bestprecision):

                bestprecision = precision

                besta = a

                bestb = b

                bestc = c



print (bestprecision, besta, bestb, bestc)
#performance of the best model 50%TRAIN 50%TEST Desc Tree

from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(criterion='gini', max_depth=besta, min_samples_split=bestb, min_samples_leaf=bestc, max_features=None, max_leaf_nodes=None)

fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)

cm = confusion_matrix(y_test, y_pre)



print ("Confusion Matrix : ")



print (cm)



print ("Accuracy : " )



print (round(accuracy_score(y_test,y_pre),3))



print (classification_report(y_test,y_pre))
#K-Folds Cross Validation using the best k and p for Desc Tree 50%Test





print ("[Train/test split] score: {:.5f}".format(clf.score(X_test, y_test)))



from sklearn.model_selection import KFold

kf = KFold (n_splits=5, random_state=4)



for train_index, test_index in kf.split (dftree) :

    print("TRAIN :", train_index, "TEST: ", test_index)

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    

for k, (train_index, test_index) in enumerate(kf.split(dftree)):

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    clf.fit(X_train, y_train)

    print ("[fold {0}] score: {1:.5f}".format(k, clf.score(X_test, y_test)))
#set train and test data 60% train 40% test



#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dftree.values, dftreetarget, test_size=0.4,random_state=0)

print (X_train)

print (X_train.shape)

print (y_train)

print (y_train.shape)

print (X_test)

print (y_test)

print (y_test.shape)
#DescTree40%Test



from sklearn.tree import DecisionTreeClassifier



for a in range (2, 5+1):

    for b in range (2, 5+1):

        for c in range (1, 5+1):

            clf = DecisionTreeClassifier(criterion='gini', max_depth=a, min_samples_split=b, min_samples_leaf=c, max_features=None, max_leaf_nodes=None)

            fit = clf.fit(X_train, y_train)

            y_pre = fit.predict(X_test)

            

            from sklearn.metrics import confusion_matrix

            from sklearn.metrics import classification_report

            cm = confusion_matrix(y_test, y_pre)

            print ("max depth = " + str(a) + " min samples split = " + str(b) + "min samples leaf" + str(c))

            print (cm)

            print (classification_report(y_test,y_pre))

            

            precision = round(float(cm.item(3))/float((cm.item(1)+cm.item(3))),2)

            if(precision > bestprecision):

                bestprecision = precision

                besta = a

                bestb = b

                bestc = c



print (bestprecision, besta, bestb, bestc)
#performance of the best model 60%TRAIN 40%TEST Desc Tree

from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(criterion='gini', max_depth=besta, min_samples_split=bestb, min_samples_leaf=bestc, max_features=None, max_leaf_nodes=None)

fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)

cm = confusion_matrix(y_test, y_pre)



print ("Confusion Matrix : ")



print (cm)



print ("Accuracy : " )



print (round(accuracy_score(y_test,y_pre),3))



print (classification_report(y_test,y_pre))
#K-Folds Cross Validation using the best k and p for Desc TREE 40%Test



print ("[Train/test split] score: {:.5f}".format(clf.score(X_test, y_test)))



from sklearn.model_selection import KFold

kf = KFold (n_splits=5, random_state=4)



for train_index, test_index in kf.split (dftree) :

    print("TRAIN :", train_index, "TEST: ", test_index)

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    

for k, (train_index, test_index) in enumerate(kf.split(dftree)):

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    clf.fit(X_train, y_train)

    print ("[fold {0}] score: {1:.5f}".format(k, clf.score(X_test, y_test)))
#set train and test data 80% train 20% test



#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dftree.values, dftreetarget, test_size=0.2,random_state=0)

print (X_train)

print (X_train.shape)

print (y_train)

print (y_train.shape)

print (X_test)

print (y_test)

print (y_test.shape)
#DescTree20%Test



from sklearn.tree import DecisionTreeClassifier



for a in range (2, 5+1):

    for b in range (2, 5+1):

        for c in range (1, 5+1):

            clf = DecisionTreeClassifier(criterion='gini', max_depth=a, min_samples_split=b, min_samples_leaf=c, max_features=None, max_leaf_nodes=None)

            fit = clf.fit(X_train, y_train)

            y_pre = fit.predict(X_test)

            

            cm = confusion_matrix(y_test, y_pre)

            print ("max depth = " + str(a) + " min samples split = " + str(b) + "min samples leaf" + str(c))

            print (cm)

            print (classification_report(y_test,y_pre))

            

            precision = round(float(cm.item(3))/float((cm.item(1)+cm.item(3))),2)

            if(precision > bestprecision):

                bestprecision = precision

                besta = a

                bestb = b

                bestc = c



print (bestprecision, besta, bestb, bestc)
#performance of the best model 80%TRAIN 20%TEST Desc Tree

from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(criterion='gini', max_depth=besta, min_samples_split=bestb, min_samples_leaf=bestc, max_features=None, max_leaf_nodes=None)

fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)

cm = confusion_matrix(y_test, y_pre)



print ("Confusion Matrix : ")



print (cm)



print ("Accuracy : " )



print (round(accuracy_score(y_test,y_pre),3))



print (classification_report(y_test,y_pre))
#K-Folds Cross Validation using the best k and p for Desc TREE 20%Test





print ("[Train/test split] score: {:.5f}".format(clf.score(X_test, y_test)))



from sklearn.model_selection import KFold

kf = KFold (n_splits=5, random_state=4)



for train_index, test_index in kf.split (dftree) :

    print("TRAIN :", train_index, "TEST: ", test_index)

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    

for k, (train_index, test_index) in enumerate(kf.split(dftree)):

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    clf.fit(X_train, y_train)

    print ("[fold {0}] score: {1:.5f}".format(k, clf.score(X_test, y_test)))
#set train and test data 50% train 50% test



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dftree.values, dftreetarget, test_size=0.5,random_state=0)

print (X_train)

print (X_train.shape)

print (y_train)

print (y_train.shape)

print (X_test)

print (y_test)

print (y_test.shape)
#KNearest Neighbour Clasification 50%



#selecting the best attribute k and p using iteration



from sklearn.neighbors import KNeighborsClassifier



bestprecision = 0



for k in range (1, 25+1):

    for i in range (1, 3+1):

        clf = KNeighborsClassifier(n_neighbors = k, weights='distance', metric='minkowski', p=i)

        fit = clf.fit(X_train, y_train)

        y_pre = fit.predict(X_test)

        from sklearn.metrics import confusion_matrix

        from sklearn.metrics import classification_report

        cm = confusion_matrix(y_test, y_pre)

        print ("k = " + str(k) + " p = " + str(i))

        print (cm)

        print (classification_report(y_test,y_pre))

        

        precision = round(float(cm.item(3))/float((cm.item(1)+cm.item(3))),2)

        if(precision > bestprecision):

            bestprecision = precision

            bestk = k

            bestp = i



print (bestprecision, bestk, bestp)
#performance of the best model 50%TRAIN 50%TEST k-NN

from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(criterion='gini', max_depth=besta, min_samples_split=bestb, min_samples_leaf=bestc, max_features=None, max_leaf_nodes=None)

fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)

cm = confusion_matrix(y_test, y_pre)



print ("Confusion Matrix : ")



print (cm)



print ("Accuracy : " )



print (round(accuracy_score(y_test,y_pre),3))



print (classification_report(y_test,y_pre))
#K-Folds Cross Validation using the best k and p for k nearest neighbors 50%Test





print ("[Train/test split] score: {:.5f}".format(clf.score(X_test, y_test)))



from sklearn.model_selection import KFold

kf = KFold (n_splits=5, random_state=4)



for train_index, test_index in kf.split (dftree) :

    print("TRAIN :", train_index, "TEST: ", test_index)

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    

for k, (train_index, test_index) in enumerate(kf.split(dftree)):

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    clf.fit(X_train, y_train)

    print ("[fold {0}] score: {1:.5f}".format(k, clf.score(X_test, y_test)))
#set train and test data 60% train 40% test



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dftree.values, dftreetarget, test_size=0.5,random_state=0)

print (X_train)

print (X_train.shape)

print (y_train)

print (y_train.shape)

print (X_test)

print (y_test)

print (y_test.shape)
#KNearest Neighbour Clasification



#selecting the best attribute k and p using iteration



#from sklearn.neighbors import KNeighborsClassifier



for k in range (1, 25+1):

    for i in range (1, 3+1):

        clf = KNeighborsClassifier(n_neighbors = k, weights='distance', metric='minkowski', p=i)

        fit = clf.fit(X_train, y_train)

        y_pre = fit.predict(X_test)

        #from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pre)

        print ("k = " + str(k) + " p = " + str(i))

        print (cm)

        print (classification_report(y_test,y_pre))

        

        precision = round(float(cm.item(3))/float((cm.item(1)+cm.item(3))),2)

        if(precision > bestprecision):

            bestprecision = precision

            bestk = k

            bestp = i



print (bestprecision, bestk, bestp)

        
#performance of the best model 50%TRAIN 40%TEST k-NN

from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(criterion='gini', max_depth=besta, min_samples_split=bestb, min_samples_leaf=bestc, max_features=None, max_leaf_nodes=None)

fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)

cm = confusion_matrix(y_test, y_pre)



print ("Confusion Matrix : ")



print (cm)



print ("Accuracy : " )



print (round(accuracy_score(y_test,y_pre),3))



print (classification_report(y_test,y_pre))
#K-Folds Cross Validation using the best k and p for k nearest neighbors





print ("[Train/test split] score: {:.5f}".format(clf.score(X_test, y_test)))



#from sklearn.model_selection import KFold

kf = KFold (n_splits=5, random_state=4)



for train_index, test_index in kf.split (dftree) :

    print("TRAIN :", train_index, "TEST: ", test_index)

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    

for k, (train_index, test_index) in enumerate(kf.split(dftree)):

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    clf.fit(X_train, y_train)

    print ("[fold {0}] score: {1:.5f}".format(k, clf.score(X_test, y_test)))
#set train and test data 80% train 20% test



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dftree.values, dftreetarget, test_size=0.5,random_state=0)

print (X_train)

print (X_train.shape)

print (y_train)

print (y_train.shape)

print (X_test)

print (y_test)

print (y_test.shape)
#KNearest Neighbour Clasification



#selecting the best attribute k and p using iteration



#from sklearn.neighbors import KNeighborsClassifier



for k in range (1, 25+1):

    for i in range (1, 3+1):

        clf = KNeighborsClassifier(n_neighbors = k, weights='distance', metric='minkowski', p=i)

        fit = clf.fit(X_train, y_train)

        y_pre = fit.predict(X_test)



        cm = confusion_matrix(y_test, y_pre)

        print ("k = " + str(k) + " p = " + str(i))

        print (cm)

        print (classification_report(y_test,y_pre))

        

        precision = round(float(cm.item(3))/float((cm.item(1)+cm.item(3))),2)

        if(precision > bestprecision):

            bestprecision = precision

            bestk = k

            bestp = i



print (bestprecision, bestk, bestp)
#performance of the best model 80%TRAIN 20%TEST k-NN

#from sklearn.metrics import accuracy_score



clf = KNeighborsClassifier(n_neighbors = bestk, weights='distance', metric='minkowski', p=bestp)

fit = clf.fit(X_train, y_train)

y_pre = fit.predict(X_test)

cm = confusion_matrix(y_test, y_pre)



print ("Confusion Matrix : ")



print (cm)



print ("Accuracy : " )



print (round(accuracy_score(y_test,y_pre),3))



print (classification_report(y_test,y_pre))
#K-Folds Cross Validation using the best k and p for k nearest neighbors 20%Test





print ("[Train/test split] score: {:.5f}".format(clf.score(X_test, y_test)))



#from sklearn.model_selection import KFold

kf = KFold (n_splits=5, random_state=4)



for train_index, test_index in kf.split (dftree) :

    print("TRAIN :", train_index, "TEST: ", test_index)

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    

for k, (train_index, test_index) in enumerate(kf.split(dftree)):

    X_train, X_test = dftree.values[train_index], dftree.values[test_index]

    y_train, y_test = dftreetarget[train_index], dftreetarget[test_index]

    clf.fit(X_train, y_train)

    print ("[fold {0}] score: {1:.5f}".format(k, clf.score(X_test, y_test)))