# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
print(train_df.columns.values)
# preview the data

train_df.head()
train_df.tail()
train_df.info()

print('_'*40)

test_df.info()


# train_df.describe()





# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.

print('SURVIVAL COLUMN DATA ANALYSIS BELOW:')

print('SURVIVAL COLUMN LENGTH(SAME AS NUMBER OF PEOPLE):')

newdf = train_df[train_df.columns[1]] 

print(len(newdf))

print('38% OF ALL PEOPLE WERE DEAD OR ALIVE?(SURVIVAL IS 1, DEAD IS 0):')

print(np.percentile(newdf, 62))





# Review Parch distribution using `percentiles=[.75, .8]`









# SibSp distribution `[.68, .69]`

print('30% OF PEOPLE TRAVELED WITH SIBLING/SPOUSE OF NOT?(0 NOT, >0 YES)');

print(' Nearly 30% of the passengers HAD siblings and/or spouse aboard: ')

dfsibsp = train_df[train_df.columns[6]] 

print(np.percentile(dfsibsp, 69))

print('PEOPLE TRAVELLED WITH HIGHEST NUMBER OF SIBLINGS HAD HOW MANY OF THEM?');

print(np.amax(dfsibsp))



# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`









# Review fared paid using percentiles = 99?

print('UNDER 1%(super rich ppl!) paid on average: ')

dfFarePaid = train_df[train_df.columns[9]] 

print(np.percentile(dfFarePaid, 99))

print('HIGHEST paid: ')

print(np.amax(dfFarePaid))

print('LOWEST(jack) paid: ')

print(np.amin(dfFarePaid))
train_df.describe(include=['O'])
print("CORELATION BETWEEN CLASS AND SURVIVAL RATE BELOW:")

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=True)

print("CONCLUSION: THE RICHER YOU WERE, THE MORE LIKELY YOU SURVIVED")
print("CORELATION BETWEEN GENDER AND SURVIVAL RATE BELOW:")

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

print("CONCLUSION: QUOTE FROM BILL BURR: IN THE UNLIKELY EVENT OF US BOTH ON TITANIC, YOU GET TO LEAVE AND I HAVE TO STAY")
print("CORELATION BETWEEN NUMBER OF SIBLING/SPOUSE AND SURVIVAL RATE BELOW:")

train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

print("CONCLUSION: SOMEHOW THE MORE SIBLINGS YOU HAVE WITH YOU THE LESS LIKELY YOU WILL SURVIVE?");
print("CORELATION BETWEEN PARTCH(?) AND SURVIVAL RATE BELOW:")

train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

print("CONCLUSION: NOT REALLY A CLEAR CORRELATION I WOULD SAY?");


print("HISTOGRAM CHART THAT SHOWS THE CORRELATION BETWEEN AGE AND SURVIVAL:");

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)

print("CONCLUSION: 1. Infants (Age <=4) had high survival rate. 2. Oldest passengers (Age = 80) survived. 3. Large number of 15-25 year olds did not survive. 4. Most passengers are in 15-35 age range.");

print("HISTOGRAM CHART THAT SHOWS THE CORRELATION BETWEEN CLASS(1-3) AND SURVIVAL(Correlating numerical and ordinal features):")

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();

print("CONCLUSION: 1.Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.2.Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.3.Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.4.Pclass varies in terms of Age distribution of passengers.");

print("DECISION: Consider Pclass for model training.");
print("HISTOGRAM CHART THAT SHOWS THE CORRELATION BETWEEN EMBARKED, GENDER, AND SURVIVAL(Correlating categorical features):")

# grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()

print("CONCLUSION: 1. Female passengers had much better survival rate than males. Confirms classifying (#1).2. Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived. 3. Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2). 4. Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).");

print("DECISION: 1. Add Sex feature to model training. 2. Complete and add Embarked feature to model training.");
print("HISTOGRAM CHART THAT SHOWS THE CORRELATION BETWEEN EMBARKED, GENDER, FARE, AND SURVIVAL(Correlating categorical and numerical features):")

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()

print("CONCLUSION: 1. Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges. 2. Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).");

print("DECISION: Consider banding Fare feature. ");

print("DROPPING FEATURES THAT ARE NOT LIKELY TO BE CORRELATED TO SURVIVAL: ")





print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

print("2 FEATURES DROPPED FOR BOTH TRAINING AND TESTING DATA SETS, where applicable we perform operations on both training and testing datasets together to stay consistent.")
print("THE FOLLOWING CODE ATTEMPTS TO EXTRACT TITLE FROM NAMES, LOOKING FOR CORRELATION BETWEEN TITLE AND SURVIVAL");

print("SINCE TITLES MOSTLY RELATED TO GENDER, IT ACTUALLY MAKES SENSE THAT GENDER AND SURVIVAL ARE SOMEWHAT CORRELATED, SUCH AS MRS ARE MORE LIKELY TO SURVIVE THAN MR, I GUESS");

print("AND WE REPLACE NAMES WITH TITLES, AND REPLACE TITLES WITH NUMBERS")

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])



print("REPLACE SOME OF THE TITLES WITH RARE, BECAUSE THEY ARE, MOSTLY LIKE WON'T CAUSE ANY TROUBLE FOR ANALYSING DATA: ")

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

 

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
print("REPLACE CATEGORICAL TITLES WITH NUMBERS(ORDINAL): ")   

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
print("DROP NAME COLUMN SINCE WE DON'T NEED IT ANYMORE, IN BOTH TRAIN AND TEST DATA SET: ")

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
print("CONVERTING MORE CATEGORICAL FEATURES TO NUMERICAL NUMBERS, WHICH IS REQUIRED FOR MODEL ALGORITHMS AND ACHIEVING FEATURE COMPLETING GOAL");

print("CONVERTING GENDER FROM F/M TO 0/1: ")

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
print("NOW ATTEMPING TO COMPLETE THE MISSING DATA FOR AGE")

print("WE GUESS THE CORRECT AGE BY USING OTHER CORRELATED DATA: GENDER AND PCLASS: ")

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
print("FIRST PREPARE AN EMPTY ARRAY FOR CALCULATED AGE BASED ON 1 of 6 GENDER + PCLASS COMBINATION:")

guess_ages = np.zeros((2,3))

guess_ages
print("THEN FOR EACH OF THE COMBINATION, CALCULATE AN AGE THAT IS GENDER + PCLASS APPROPRIATE(DON'T KNOW HOW ELSE TO PUT IT)")

print("AND FOR EACH COMBINATION, FIND IN THE DATASET THAT HAS AGE VALUE MISSING FOR THAT COMBINATION, AND REPLACE MISSING VALUE WITH CALCULATED AGE:")

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()

print("AFTER ADDING MISSING DATA, WE GROUP AGE TO BANDS(GROUPS) AND INVESTIGAE THE CORRELATION BETWEEN AGE BANDS AND SURVIVAL: ")

print("NOTE1: NOT SURE HOW PD.CUT WORKS BELOW, IT SEEMS THAT THE SECOND PARAMETER DETERMINES HOW MANY BANDS YOU CAN GET: ");

print("NOTE2: MAYBE GROUP AGES CAN ALSO PREVENT THE FINAL MODEL FROM OVERFITTING?")

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
print("REPLACE AGES WITH ORDINALS USING THE OBSERVATION ABOVE, DO IT IN BOTH TRAIN AND TEST DATASET: ")

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
print("AFTER REPLACING AGES WITH ORDINAL VALUES, WE CAN REMOVE THE AGE BAND FEATURE:")

print("NOTE: THE ORIGINAL POST SAYS WE CAN NOT REMOVE IT?");

train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
print("THE CORRELATIONS BETWEEN PARCH/SIBSPO AND SURVIVAL ARE NOT THAT OBVIOUS")

print("SO INSTEAD OF USING SIBLING/PARENTS etc TO BUILD THE MODEL, WE WANT TO USE IF THAT PERSON IS ALONE OR NOT")

print("TO DO THAT, WE FIRST BUILD A COLUMN FAMILYSIZE, WHICH IS THE SUM OF PARCH AND SIBSPO: ")

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("AND WE USE THE FAMILY SIZE TO CHECK IF SOMEONE WAS ALONE ON THE SHIP, AND SEE THE CORRELATION BETWEEN LONELINESS AND SURVIVAL: ")

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
print("SO IT SEEMS THERE IS A CORRELATION: YOU WERE MORE LIKELY TO SURVIVE IF ALONE");

print("THEREFORE INSTEAD OF USING SIBSPO/PARCH, WE USE ISALONE TO BUILD OUR MODEL")

print("DROP SIBSPO/PARCH/FAMILYSIZE SINCE WE NO LONGER USE THEM: ")

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
print("WIERD FEATURE CREATED IN ORIGINAL TUTORIAL: AGE * PCLASS")

print("I HAVE NO IDEA WHY CREATING IT CAN HELP, JUST FOLLOWING THE STEPS HERE:")



for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
print("2 MISSING VALUES FROM EMBARKED FEATURE, FILL THEM UP WITH MOST COMMON VALUE");

print("FIRST FIND THE MOST COMMON VALUE: ")

freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
print("AFTER FINDING MOST COMMON VALUE, FILL THE MISSING VALUES UP WITH IT: ")

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print("AND AS WITH OTHER CATEGORICAL VALUES, EMBARKED VALUES ARE CONVERTED TO NUMERICAL VALUES, WHICH IS REQUIRED FOR BUILDING MODEL: ")

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
print("FILL THE ONLY MISSING VALUE FROM FARE WITH MEDIAN VALUE OF ALL FARES: ")

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
print("AFTER FILLING THE MISSING VALUE, CREATE BANDS FOR FARES: ")

print("NOTE: 1. I'M NOT SURE WHY 4 BANDS, JUST FOLLOWING STEPS 2. THE ORIGINAL TUTORIAL SAYS WE CAN NOT(IS IT NOW?) CREATE BANDS?")

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
print("REPLACE FARES WITH BANDS GENERATED, REMOVE THE FAREBAND COLUMN: ")



for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)

test_df.head(10)

print("FIRST GET TRAINING DATA AND TRAINING LABELS, ALSO DROP THE PASSENGERID FOR TEST SINCE NO USE FOR THAT:")

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression

print("USE LOGISTIC REGRESSION TO PREDICT WITH TEST DATA, ACCURACY: ")

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
print("CORRELATION BETWEEN SURVIVAL AND ALL THE FEATURES: ")

print("CONCLUSION:")

print("1. GENDER IS STRONGLY POSITIVELY CORRELATED TO SURVIVAL 2. TITLE, WHICH IS RELATED TO GENDER, IS ALSO POSITIVELY RELATED TO SURVIVAL");

print("3. THE RELATIONSHIP BETWEEN NUBMER OF FAMILY AND SURVIVAL IS STILL NOT THAT CLEAR EVEN AFTER COMBINING THEM INTO ON FEATURE 4. THE POORER YOU ARE, THE MORE LIKELY YOU WILL DIE")

coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines

print("USE SVC(ISN'T IT SUPPOSED TO BE SVM?) TO PREDICT WITH TEST DATA, ACCURACY: ")

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
# k-Nearest Neighbors algorithm 

print("USE KNN TO PREDICT WITH TEST DATA, ACCURACY: ")

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes

print("USE Gaussian Naive Bayes TO PREDICT WITH TEST DATA, ACCURACY: ")

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron

print("USE Perceptron TO PREDICT WITH TEST DATA, ACCURACY: ")

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC

print("USE SVC TO PREDICT WITH TEST DATA, ACCURACY: ")

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent

print("USE SGD TO PREDICT WITH TEST DATA, ACCURACY: ")

sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree

print("USE Decision Tree TO PREDICT WITH TEST DATA, ACCURACY: ")

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest

print("USE Random Forest TO PREDICT WITH TEST DATA, ACCURACY: ")

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
print("MODEL EVALUATION: ")

print("QUOTE FROM ORIGINAL AUTHOR: We can now rank our evaluation of all the models to choose the best one for our problem. While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set.")

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
print("SUBMISSION OF RESULT(TRY SUBMIT): ")

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)