import numpy as np

import pandas as pd

from matplotlib import pyplot

from sklearn.ensemble import RandomForestClassifier 

from collections import OrderedDict



# Read data from the respective datasets

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# Get an idea of the data types and number of non-null entries

print(train.info()) 
# Create a new data frame that contains cleaned up features

train2 = train

test2 = test



# Convert 'male' and 'female' categorical values in the 'Sex' category

# into 1 and 0 integers, respectively, in a new 'Gender' category

train2['Gender'] = train2.Sex.map( {'female':0,'male':1} ).astype(int)

test2['Gender'] = test2.Sex.map( {'female':0,'male':1} ).astype(int)
# Since there are some non-null 'Age' entries, it's 

# useful to fill the remaining null entries with some estimated values 

# so the data points associated with those entries can be used for training

train2['Age2'] = train2.Age

test2['Age2'] = test2.Age



# The estimated age values will be resampled using bootstrapping from

# relevant (Pclass, Gender) set

tot_nulls = 0

tot_notnulls = 0

for pclass in range(1,4):

	for gender in range(0,2):

		####

		# The following are for imputing ages for training data

		# Create a variable that holds all non-null Age values for

		# each (Pclass, Gender) combination

		age_array = train2[ (train2.Pclass == pclass) & \

			(train2.Gender == gender) & (train2.Age.notnull()) ].Age.values

		# Determine the number of null Age values for each

		# (Pclass, Gender) combination

		num_nulls = len( train2[ (train2.Pclass == pclass) & \

			(train2.Gender == gender) & (train2.Age.isnull()) ].Age )

		# Resample the Age values with replacement

		imputed_ages = age_array[np.random.choice(len(age_array),num_nulls)]

		# Now assign the imputed age values to each (Pclass, Gender) combo

		train2.loc[ (train2.Pclass == pclass) & \

			(train2.Gender == gender) & (train2.Age.isnull()), 'Age2' ] = imputed_ages



		# Let's check the histogram of the original Age distribution vs. 

		# the filled Age2 distribution as a sanity check

		pyplot.title('(Pclass=%d, Gender=%d)'%(pclass,gender))

		pyplot.hist(age_array, bins=20, color='blue', alpha=0.5, normed=True, label='Original Age')

		pyplot.hist(np.concatenate((age_array,imputed_ages)), bins=20, color='red', alpha=0.5, normed=True, label='Imputed Age')

		pyplot.legend(loc='upper right')

		pyplot.show()

		

		####

		# The following are for imputing ages for test data

		# Create a variable that holds all non-null Age values for

		# each (Pclass, Gender) combination

		age_array = test2[ (test2.Pclass == pclass) & \

			(test2.Gender == gender) & (test2.Age.notnull()) ].Age.values

		# Determine the number of null Age values for each

		# (Pclass, Gender) combination

		num_nulls = len( test2[ (test2.Pclass == pclass) & \

			(test2.Gender == gender) & (test2.Age.isnull()) ].Age )

		# Resample the Age values with replacement

		imputed_ages = age_array[np.random.choice(len(age_array),num_nulls)]

		# Now assign the imputed age values to each (Pclass, Gender) combo

		test2.loc[ (test2.Pclass == pclass) & \

			(test2.Gender == gender) & (test2.Age.isnull()), 'Age2' ] = imputed_ages



		# Let's check the histogram of the original Age distribution vs. 

		# the filled Age2 distribution as a sanity check

		#pyplot.title('(Pclass=%d, Gender=%d)'%(pclass,gender))

		#pyplot.hist(age_array, bins=20, color='blue', alpha=0.5, normed=True, label='Original Age')

		#pyplot.hist(np.concatenate((age_array,imputed_ages)), bins=20, color='red', alpha=0.5, normed=True, label='Imputed Age')

		#pyplot.legend(loc='upper right')

		#pyplot.show()
# Check to see that all the null age values are filled in

print(train2.info())

print(test2.info())



# The test data has a null entry for Fare.  Let's fill it in with the mean

test2.loc[ test2.Fare.isnull(), 'Fare' ] = test2.Fare.mean()
# Let's create a few features

# The first feature will be a sum of SibSp and Parch

train2['Relatives'] = train2.SibSp + train2.Parch

test2['Relatives'] = test2.SibSp + test2.Parch
# Now let's drop all features that have categorical or irrelevant data

train2 = train2.drop(['PassengerId','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

test2 = test2.drop(['PassengerId','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)



# Convert dataframe to ndarray

train_data = train2.values

test_data = test2.values



X = train_data[0::,1::] # features input

y = train_data[0::,0] # target output
########################################################################

# Let's train a random forest estimator

#

# Create the random forest object which will include all the parameters

# for the fit

#

# For the random forests, the number of trees, n_estimators, and maximum

# number of random features to use at each split, max_features, are the

# most important hyper-parameters that affect a forest's prediction ability.

# I will use 100 <= n_estimators <= 500 and

# max_features = {'sqrt', n_features/2, None}

#

# I will also set the following:

# criterion = 'entropy' simply because I favor the information-theoretic error function,

# although its performance is very similar to 'gini' criterion

#

# oob_score = True because I want to use it as a proxy for test error

#

# warm_start = True so it speeds up learning over the ranges of n_estimators

########################################################################

if True:

	RANDOM_STATE = 1234

	

	# NOTE: Setting the `warm_start` construction parameter to `True` disables

	# support for parallelized ensembles but is necessary for tracking the OOB

	# error trajectory during training.

	ensemble_clfs = [

	    ("RandomForestClassifier, max_features='sqrt'",

	        RandomForestClassifier(warm_start=True, max_features="sqrt",

								   oob_score=True, criterion='gini',                               

	                               random_state=RANDOM_STATE)),

	    ("RandomForestClassifier, max_features=n_features/2",

	        RandomForestClassifier(warm_start=True, max_features=int(len(train_data[0,1::])/2),

	                               oob_score=True, criterion='gini',

	                               random_state=RANDOM_STATE)),

	    ("RandomForestClassifier, max_features=None",

	        RandomForestClassifier(warm_start=True, max_features=None,

	                               oob_score=True, criterion='gini',

	                               random_state=RANDOM_STATE))

	]

	

	# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.

	error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

	

	# Range of `n_estimators` values to explore.

	min_estimators = 15

	max_estimators = 200

	

	for label, clf in ensemble_clfs:

	    for i in range(min_estimators, max_estimators + 1):

	        clf.set_params(n_estimators=i)

	        clf.fit(X, y)

	

			# Record the OOB error for each `n_estimators=i` setting.

	        oob_error = 1 - clf.oob_score_

	        error_rate[label].append((i, oob_error))

	

	# Generate the "OOB error rate" vs. "n_estimators" plot.

	for label, clf_err in error_rate.items():

	    xs, ys = zip(*clf_err)

	    pyplot.plot(xs, ys, label=label)

	

	pyplot.xlim(min_estimators, max_estimators)

	pyplot.xlabel("n_estimators")

	pyplot.ylabel("OOB error rate")

	pyplot.legend(loc="upper right")

	pyplot.show()
########################################################################

# Let's train a random forest estimator



# From the model-selection performed above, it appears the following

# random forest configuration performs relatively well



forest = RandomForestClassifier(n_estimators=200, criterion='gini',

							   max_features=int(len(train_data[0,1::])/2))

forest.fit(X,y) 



output = forest.predict(test_data) 

output = output.astype(int)



# Write the output to a csv with PassengerId and Survived columns

out_data = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':output})

out_data.to_csv('Submission_Titanic_Pt1_2017.csv',index=False)