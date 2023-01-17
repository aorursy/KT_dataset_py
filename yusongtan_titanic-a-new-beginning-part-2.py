import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import re



# Read data from the respective datasets

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# Get a quick view of the data

print(train.info())

print('-'*50)



# Create a copy of train and test

train2 = train

test2 = test



# Drop Cabin and Ticket features: no strong evidential way to impute or use them

train2 = train2.drop(['Cabin', 'Ticket'], axis=1)

test2 = test2.drop(['Cabin', 'Ticket'], axis=1)
# Convert 'male' and 'female' categorical values in the 'Sex' category

# into 1 and 0 integers, respectively, in a new 'Gender' category

train2['Gender'] = train2.Sex.map( {'female':0,'male':1} ).astype(int)

test2['Gender'] = test2.Sex.map( {'female':0,'male':1} ).astype(int)
# Also create a 'Title' feature

def get_title(name):

	title = re.search(' ([A-Za-z]+)\.', name)

	if title:

		return title.group(1)

	else:

		return ''

			

titles = train2.Name.apply(get_title)

print(pd.value_counts(titles))

print('-'*50)

train2['Title'] = titles



# Convert some titles to existing categories

train2.loc[train2.Title=='Mlle','Title'] = 'Miss'

train2.loc[train2.Title=='Ms','Title'] = 'Miss'

train2.loc[train2.Title=='Mme','Title'] = 'Miss'



# Create a rare-title category 

rare_titles = ['Dr','Rev','Col','Major','Countess','Lady','Jonkheer','Don',

			'Capt','Sir']

for rt in rare_titles:

	train2.loc[train2.Title==rt,'Title'] = 'Rare_title'

	

titles = test2.Name.apply(get_title)

print(pd.value_counts(titles))

print('-'*50)

test2['Title'] = titles



# Convert some titles to existing categories

test2.loc[test2.Title=='Ms','Title'] = 'Miss'



rare_titles = ['Dr','Rev','Col','Dona']

for rt in rare_titles:

	test2.loc[test2.Title==rt,'Title'] = 'Rare_title'
# Get a dataset without missing Age samples, so we can do some preliminary analysis

train_nonull = train2.dropna()



# Determine how Survived correlates against Age by fitting a linear regression

sns.lmplot(x='Age', y='Survived', data=train_nonull, y_jitter=0.05)

plt.subplots_adjust(top=0.9)

sns.plt.suptitle('Correlation between Survived and Age')



# Now see how the correlation between Surived and Age, conditioned on Pclass

sns.lmplot(x='Age', y='Survived', hue='Pclass', data=train_nonull, 

		y_jitter=0.05)

plt.subplots_adjust(top=0.9)

sns.plt.suptitle('Correlation between Survived and Age, Given Pclass')



# Now see how the correlation between Surived and Age, conditioned on Sex and

# Pclass

sns.lmplot(x='Age', y='Survived', hue='Pclass', col='Sex', data=train_nonull,

		y_jitter=0.05)

plt.subplots_adjust(top=0.9)

sns.plt.suptitle('Correlation between Survived and Age, Given Pclass and Sex')
# Determine how Survived correlates against Pclass

sns.lmplot(x='Pclass', y='Survived', data=train2, y_jitter=0.05,

		x_jitter=0.05)

plt.subplots_adjust(top=0.9)

sns.plt.suptitle('Correlation between Survived and Pclass')
# Determine how Survived correlates against Gender

sns.lmplot(x='Gender', y='Survived', data=train2, y_jitter=0.05,

		x_jitter=0.05)

plt.subplots_adjust(top=0.9)

sns.plt.suptitle('Correlation between Survived and Gender')
# Now see how the correlation between Surived and SibSp

sns.lmplot(x='SibSp', y='Survived', data=train2, y_jitter=0.05,

		x_jitter=0.05)

plt.subplots_adjust(top=0.9)

sns.plt.suptitle('Correlation between Survived and SibSp')



# Now see how the correlation between Surived and Parch

sns.lmplot(x='Parch', y='Survived', data=train2, y_jitter=0.05,

		x_jitter=0.05)

plt.subplots_adjust(top=0.9)

sns.plt.suptitle('Correlation between Survived and Parch')



# How does Survived correlate against total relatives?

train2['Relatives'] = train2.Parch + train2.SibSp

test2['Relatives'] = test2.Parch + test2.SibSp

sns.lmplot(x='Relatives', y='Survived', data=train2, y_jitter=0.05,

		x_jitter=0.05)

plt.subplots_adjust(top=0.9)

sns.plt.suptitle('Correlation between Survived and Relatives')
# How does Fare correlate against Survived

sns.lmplot(x='Fare', y='Survived', data=train2, y_jitter=0.05)

plt.subplots_adjust(top=0.9)

sns.plt.suptitle('Correlation between Survived and Fare')



# How does Fare correlate against Survived, given Pclass and Sex?

g = sns.lmplot(x='Fare', y='Survived', hue='Pclass', col='Sex', data=train2,

			y_jitter=0.05)

plt.subplots_adjust(top=0.9)

plt.suptitle('Correlation between Survived and Fare, Given Pclass and Sex')

g = g.set(ylim=(-2, 2))
# There are two missing Embarked values.  Let's just randomly assign them

train2.loc[train2.Embarked.isnull(),'Embarked'] = ['S', 'S']



# How does Embarked correlated against Survived?

plt.figure()

sns.countplot(x='Embarked', hue='Survived', order=['S','C','Q'], 

		data=train2)

plt.subplots_adjust(top=0.9)

plt.suptitle('Correlation between Survived and Embarked')



# How does Embarked correlated against Survived, given Pclass==1

plt.figure()

sns.countplot(x='Embarked', hue='Survived', order=['S','C','Q'],

		data=train2[train2.Pclass==1])

plt.subplots_adjust(top=0.9)

plt.suptitle('Correlation between Survived and Embarked, Given Pclass==1')



# How does Embarked correlated against Survived, given Pclass==2

plt.figure()

sns.countplot(x='Embarked', hue='Survived', order=['S','C','Q'],

		data=train2[train2.Pclass==2])

plt.subplots_adjust(top=0.9)

plt.suptitle('Correlation between Survived and Embarked, Given Pclass==2')



# How does Embarked correlated against Survived, given Pclass==3

plt.figure()

sns.countplot(x='Embarked', hue='Survived', order=['S','C','Q'],

		data=train2[train2.Pclass==3])

plt.subplots_adjust(top=0.9)

plt.suptitle('Correlation between Survived and Embarked, Given Pclass==3')
# How does Title correlate against Survived

plt.figure()

sns.countplot(x='Title', hue='Survived', data=train2)

plt.subplots_adjust(top=0.9)

plt.suptitle('Correlation between Survived and Title')
# Since there are some non-null 'Age' entries, it's 

# useful to fill the remaining null entries with some estimated values 

# so the data points associated with those entries can be used for training

train2['Age2'] = train2.Age

test2['Age2'] = test2.Age



# The estimated age values will be resampled using bootstrapping from

# relevant (Pclass, Title) set

comb_data = pd.concat([train2, test2], join='inner')

for pclass in range(1,4):

	for title in ['Mr','Miss','Mrs','Master','Rare_title']:

		####

		# The following are for imputating ages for training and test data

		# Create a variable that holds all non-null Age values for

		# each (Pclass, Title) combination

		age_array = comb_data[ (comb_data.Pclass == pclass) & 

						(comb_data.Title == title) & 

						(comb_data.Age.notnull()) ].Age.values

		if age_array.size > 0:

			# Determine the number of null Age values for each

			# (Pclass, Gender) combination

			num_nulls_train = len( train2[ (train2.Pclass == pclass) & 

							(train2.Title == title) & 

							(train2.Age.isnull()) ].Age )

			num_nulls_test = len( test2[ (test2.Pclass == pclass) & 

							(test2.Title == title) & 

							(test2.Age.isnull()) ].Age )				

			# Resample the Age values with replacement

			imputed_ages_train = np.random.choice(age_array,num_nulls_train)

			imputed_ages_test = np.random.choice(age_array,num_nulls_test)

			# Now assign the imputed age values to each (Pclass, Gender) combo

			train2.loc[ (train2.Pclass == pclass) & 

					(train2.Title == title) & 

					(train2.Age.isnull()), 'Age2' ] = imputed_ages_train

			test2.loc[ (test2.Pclass == pclass) & 

					(test2.Title == title) & 

					(test2.Age.isnull()), 'Age2' ] = imputed_ages_test



		# Let's check the histogram of the original Age distribution vs. 

		# the filled Age2 distribution as a sanity check

#		plt.figure()

#		plt.title('(Pclass=%d, Title=%s)'%(pclass,title))

#		plt.hist(age_array, bins=20, color='blue', alpha=0.5, 

#				normed=True, label='Original Age')

#		plt.hist(np.concatenate((age_array,imputed_ages_train)), 

#				bins=20, color='red', alpha=0.5, normed=True, 

#				label='Imputed Age')

#		plt.legend(loc='upper right')

#		plt.show()



# Impute missing Fare values in test

test2.loc[test2.Fare.isnull(),'Fare'] = test2.Fare.mean()
# Encode the categorical features to numerics

categories = ['Embarked','Title']



for cat in categories:

	a = pd.get_dummies(train2[cat]) # one-hot-encoding

	train2 = pd.concat([train2, a], axis=1)

	a = pd.get_dummies(test2[cat])

	test2 = pd.concat([test2, a], axis=1)



# Normalize the features that have significantly different scaling

fscale = preprocessing.StandardScaler()

categories = ['Fare','Age2']

for cat in categories:

    fscale.fit(train2[cat].values.reshape(-1,1))

    train2[cat] = fscale.transform(train2[cat].values.reshape(-1,1))

    test2[cat] = fscale.transform(test2[cat].values.reshape(-1,1))
# Drop features that we are not going to use

train2 = train2.drop(['PassengerId', 'Name', 'Sex', 'Age',

				'Embarked','Title'], axis=1)

test2 = test2.drop(['PassengerId', 'Name', 'Sex', 'Age',

				'Embarked','Title'], axis=1)
# A more concise way to look at correlation

train_corr = train2.corr().Survived.abs()

train_corr.sort_values(ascending=False, inplace=True)

print(train_corr)

print('-'*50)



# Convert data.frame to ndarray

train_data = train2.values

test_data = test2.values



X = train_data[0::,1::] # features input

y = train_data[0::,0] # target output