import tensorflow as tf

import numpy as np

import pandas as pd

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train = train.drop(['PassengerId', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis = 1)

test = test.drop(['Parch', 'Ticket', 'Cabin', 'Embarked'], axis = 1)

test.describe(include="all")
# create a combined group of both datasets

combine = [train, test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

train.describe(include="all")
#sort the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult

miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student

mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult

master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby

royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult

rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult



age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}



for x in range(len(train["AgeGroup"])):

    if train["AgeGroup"][x] == "Unknown":

        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

        

for x in range(len(test["AgeGroup"])):

    if test["AgeGroup"][x] == "Unknown":

        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]

        

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)



train.head()



#dropping the Age feature for now, might change

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

        

#map Fare values into groups of numerical values

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
test.describe(include="all")
train = train.drop(["Title", "Name"], axis = 1)

test = test.drop(["Title", "Name"], axis = 1)
sex_mapping = {'male': 0, 'female': 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
X = tf.placeholder(np.float32, [None, 5])

Y = tf.placeholder(np.float32, [None, 1])



def xavier_init(n_inputs, n_outputs, uniform=True):

    if uniform:

        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))

        return tf.random_uniform_initializer(-init_range, init_range)

    else:

        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))

        return tf.truncated_normal_initializer(stddev=stddev)



W1 = tf.get_variable("W1", shape=[5, 40], initializer=xavier_init(5, 40))

W2 = tf.get_variable("W2", shape=[40, 40], initializer=xavier_init(40, 40))

W3 = tf.get_variable("W3", shape=[40, 1], initializer=xavier_init(40, 1))



B1 = tf.Variable(tf.random_normal([40]))

B2 = tf.Variable(tf.random_normal([40]))

B3 = tf.Variable(tf.random_normal([1]))



keep_prob = tf.placeholder(tf.float32)



L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))

L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))

L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

hypothesis = tf.add(tf.matmul(L2, W3), B3)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=Y))
learning_rate = 0.0005



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



init = tf.initialize_all_variables()
trainY = pd.DataFrame(train["Survived"])
train = train.drop(["Survived"], axis = 1)
trainx = np.array(train, dtype=np.float32)

trainy = np.array(trainY, dtype=np.float32)



print(trainy.dtype)
predict = (tf.nn.sigmoid(hypothesis) > 0.5)

correct_prediction = tf.equal(predict, (Y > 0.5))



accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



testx = test.drop(["PassengerId"], axis = 1)

testx = np.array(testx, dtype = np.float32)
sess = tf.Session()

sess.run(init)

feed = {X: trainx, Y: trainy, keep_prob: 0.8}

    

for step in range(10001):

    sess.run(optimizer, feed_dict=feed)

        

    if step % 100 == 0:

        print(step, sess.run([cost, accuracy], feed_dict=feed))
predicted = sess.run(predict, feed_dict={X: testx, keep_prob: 0.8})



sol = pd.DataFrame()

sol['PassengerId'] = test['PassengerId']

sol['Survived'] = pd.Series(predicted.reshape(-1)).map({True:1, False:0})

print(sol)

sol.to_csv('solution3.csv', index=False)
ans = pd.read_csv("../input/gender_submission.csv")

acc_array = np.where(sol["Survived"] == ans["Survived"], 1, 0)

print(np.mean(acc_array, axis = 0))