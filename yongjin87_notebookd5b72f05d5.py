# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from sklearn.utils import shuffle

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# Load data files

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age":np.float64},)

test_df = pd.read_csv("../input/test.csv", dtype={"Age":np.float64},)



# Preview the data

#titanic_df.head()

test_df.head()

#titanic_df.info()

#titanic_df.describe()

#test_df.describe()
# Pre processing

# Remove unused columns

titanic_df = titanic_df.drop(["PassengerId", "Ticket", "Cabin"], axis=1)

test_df = test_df.drop(["Ticket", "Cabin"], axis=1)

#titanic_df = titanic_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

#test_df = test_df.drop(["Name", "Ticket", "Cabin"], axis=1)

#titanic_df = titanic_df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare"], axis=1)

#test_df = test_df.drop(["Name", "Ticket", "Cabin", "Fare"], axis=1)



#Name

def get_name_len(a):

    return len(a)

titanic_df["NameLen"] = titanic_df["Name"].apply(get_name_len)

test_df["NameLen"] = test_df["Name"].apply(get_name_len)

titanic_df.drop(["Name"], axis=1, inplace=True)

test_df.drop(["Name"], axis=1, inplace=True)

#Sex column

#def get_set(a):

#    return 0 if a == "male" else 1

#titanic_df["Sex"] = titanic_df["Sex"].apply(get_set)

#test_df["Sex"] = test_df["Sex"].apply(get_set)

def get_class(b):

    pclass,sex = b

    if sex == 'male':

        if pclass == 1:

            return 0

        elif pclass == 2:

            return 1

        else:

            return 2

    else:

        if pclass == 1:

            return 3

        elif pclass == 2:

            return 4

        else:

            return 5

        

titanic_df["ClassSex"] = titanic_df[["Pclass", "Sex"]].apply(get_class,axis=1)

test_df["ClassSex"] = test_df[["Pclass", "Sex"]].apply(get_class,axis=1)

titanic_df = titanic_df.drop(["Pclass", "Sex"], axis=1)

test_df = test_df.drop(["Pclass", "Sex"], axis=1)

# Age

median_age = titanic_df["Age"].dropna().median()

titanic_df["Age"][np.isnan(titanic_df["Age"])] = median_age

test_df["Age"][np.isnan(test_df["Age"])] = median_age

titanic_df["Age"] = titanic_df["Age"].astype(int)

test_df["Age"] = test_df["Age"].astype(int)

# Fare

test_df["Fare"].fillna(titanic_df["Fare"].median(), inplace=True)

titanic_df["Fare"] = titanic_df["Fare"].astype(int)

test_df["Fare"] = test_df["Fare"].astype(int)

# Embarked

def get_embarked(a):

    if a == "S":

        result = 0

    elif a == "C":

        result = 1

    elif a == "Q":

        result = 2

    return result

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

titanic_df["Embarked"] = titanic_df["Embarked"].apply(get_embarked)

test_df["Embarked"] = test_df["Embarked"].fillna("S")

test_df["Embarked"] = test_df["Embarked"].apply(get_embarked)

#embark_dummy = pd.get_dummies(titanic_df["Embarked"])

#titanic_df = titanic_df.join(embark_dummy)

#titanic_df.drop(["Embarked"], axis=1, inplace=True)

#embark_dummy_test = pd.get_dummies(test_df["Embarked"])

#test_df = test_df.join(embark_dummy_test)

#test_df.drop(["Embarked"], axis=1, inplace=True)



#print(titanic_df)
#titanic_df.describe()

test_df.describe()
X = titanic_df.drop("Survived",axis=1)

Y = titanic_df["Survived"]

X = X.as_matrix().astype(np.float32)

Y = Y.as_matrix().astype(np.float32)

#X = X.as_matrix().astype(np.int)

#Y = Y.as_matrix().astype(np.int)

X, Y = shuffle(X, Y)

print(type(X), X.shape, Y.shape)

# Normalization

mu = X.mean(axis=0)

std = X.std(axis=0)

X = (X - mu) / std



X_train = X[:-100,]

Y_train = Y[:-100]

X_valid = X[-100:,]

Y_valid = Y[-100:]



X_test = test_df.drop("PassengerId", axis=1)

#X_test = X_test.as_matrix().astype(np.float32)

# Normalization

mu = X_test.mean(axis=0)

std = X_test.std(axis=0)

X_test = (X_test - mu) / std

#print(X_train)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

#Y_pred = logreg.predict(X_test)

#logreg.score(X_train, Y_train)

logreg.score(X_valid, Y_valid)
# Random Forests - Reference



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

#Y_pred = random_forest.predict(X_test)

#random_forest.score(X_train, Y_train)

#Y_test_pred = random_forest.predict(X_test)

random_forest.score(X_valid, Y_valid)
# XGBoost

import xgboost as xgb



dtrain = xgb.DMatrix(X_train, label=Y_train)

dtest = xgb.DMatrix(X_valid, label=Y_valid)



param = {'bst:max_depth':3, 'bst:eta':0.3, 'silent':1, 'objective':'binary:logistic'}

#param = {'bst:max_depth':6, 'bst:eta':0.1, 'silent':1, 'objective':'multi:softmax'}

param['nthread'] = 4

param['eval_metric'] = 'auc'



evallist = [(dtest, 'eval'), (dtrain,'train')]



num_round = 20

bst = xgb.train(param, dtrain, num_round, evallist)

pred = bst.predict(dtest)

Y_pred = [round(value) for value in pred]

print (np.mean(Y_valid != Y_pred))

#print (Y_pred)

#print (Y_valid)
def error_rate(targets, predictions):

    return np.mean(targets != predictions)



def sigmoid(A):

    return 1 / (1 + np.exp(-A))
# Logistic regression model



class LogisticModel(object):

    def __init__(self):

        pass

    

    def fit(self, X, Y, learning_rate = 10e-6, reg=1*10e-22, epochs = 100000):

        N, D = X.shape

        

        #self.W = np.random.randn(D) / np.sqrt(D)

        #self.b = 0

        ones = np.array([[1]*N]).T

        Xb = np.concatenate((ones, X), axis=1)

        self.W = np.random.randn(D+1)

        

        for i in range(epochs):

            pY = self.forward(Xb)

            self.W -= learning_rate*(Xb.T.dot(pY-Y)+reg*self.W)

        

        '''

        for i in range(epochs):

            pY = self.forward(X)

            self.W -= learning_rate*(X.T.dot(pY-Y)+reg*self.W)

            self.b -= learning_rate*((pY-Y).sum()+reg*self.b)

            

            if i%100 == 0:

                error = self.score(X, Y)

                #print("i : ", i, "W : ", self.W, "b : ", self.b, "error : ", error)

                #print("i : ", i, "error : ", error)

        '''

        #print()

        

    def forward(self, X):

        #return sigmoid(X.dot(self.W) + self.b + 0.5*(self.W**2).sum() + 0.5*(self.b*2))

        #return sigmoid(X.dot(self.W) + self.b)

        return sigmoid(X.dot(self.W))

    

    def predict(self, X):

        N, D = X.shape

        ones = np.array([[1]*N]).T

        Xb = np.concatenate((ones, X), axis=1)

        pY = self.forward(Xb)

        return np.round(pY)

    

    def score(self, X, Y):

        #N, D = X.shape

        #ones = np.array([[1]*N]).T

        #Xb = np.concatenate((ones, X), axis=1)

        prediction = self.predict(X)

        return 1-error_rate(Y, prediction)

        

model = LogisticModel()

model.fit (X_train, Y_train)

error = model.score(X_train, Y_train)

Y_pred = model.predict(X_test)

error_valid = model.score(X_valid, Y_valid)

print("Train Error : ", error)

print("Valie Error : ", error_valid)


# Make Y train indication matrix

N = len(Y_train)

Y_train_ind = np.zeros((N, 2))

for i in range(N):

    Y_train_ind[i, Y_train[i]] = 1

    

# Make Y valid indication matrix

N = len(Y_valid)

Y_valid_ind = np.zeros((N, 2))

for i in range(N):

    Y_valid_ind[i, Y_valid[i]] = 1



#print (X_train)

print (X_train.shape, Y_train_ind.shape, X_test.shape)

#Y_test = test_df["Survived"]



# Set basic variables

max_iter = 500

print_period = 10

lr = 10e-6

reg = 1*10e-10



N, D = X_train.shape

print ("N : ", N, "D : ", D)



# hidden layers

#M1 = 1000

#M2 = 1000

M1 = 300

M2 = 300

K = 2



#w1_init = np.random.randn(D,M1) / np.sqrt(D+M1)

w1_init = np.random.randn(D,M1) 

b1_init = np.zeros(M1)

#w2_init = np.random.randn(M1,M2) / np.sqrt(M1+M2)

w2_init = np.random.randn(M1,M2)

b2_init = np.zeros(M2)

#w3_init = np.random.randn(M2,K) / np.sqrt(M2+K)

w3_init = np.random.randn(M2,K) 

b3_init = np.zeros(K)



# Define tensorflow variables

X = tf.placeholder(tf.float32, shape=(None, D), name='X')

T = tf.placeholder(tf.float32, shape=(None, K), name='T')

W1 = tf.Variable(w1_init.astype(np.float32))

b1 = tf.Variable(b1_init.astype(np.float32))

W2 = tf.Variable(w2_init.astype(np.float32))

b2 = tf.Variable(b2_init.astype(np.float32))

W3 = tf.Variable(w3_init.astype(np.float32))

b3 = tf.Variable(b3_init.astype(np.float32))





#Z1 = tf.nn.relu(tf.matmul(X,W1) + b1)

#Z2 = tf.nn.relu(tf.matmul(Z1,W2) + b2)

#Z1 = tf.sigmoid(tf.matmul(X,W1) + b1)

#Z2 = tf.sigmoid(tf.matmul(Z1,W2) + b2)

Z1 = tf.tanh(tf.matmul(X,W1) + b1)

Z2 = tf.tanh(tf.matmul(Z1,W2) + b2)

Y = tf.matmul(Z2, W3) + b3

'''

# Drop out version

X_drop = tf.nn.dropout(X, 0.8)

Z1 = tf.nn.relu(tf.matmul(X_drop,W1) + b1)

Z1_drop = tf.nn.dropout(Z1, 0.5)

Z2 = tf.nn.relu(tf.matmul(Z1_drop,W2) + b2)

Z2_drop = tf.nn.dropout(Z2, 0.5)

Y = tf.matmul(Z2_drop, W3) + b3



X_pred = X * 0.8

Z1_pred = tf.nn.relu(tf.matmul(X_pred,W1) + b1)

Z1_pred_drop = Z1_pred * 0.5

Z2_pred = tf.nn.relu(tf.matmul(Z1_pred_drop,W2) + b2)

Z2_pred_drop = Z2_pred * 0.5

Y_pred = tf.matmul(Z2_pred_drop, W3) + b3

'''





params = [W1, b1, W2, b2, W3, b3]

rcost = reg*sum([tf.nn.l2_loss(p) for p in params])

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Y, T)) + rcost



#train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

train_op = tf.train.RMSPropOptimizer(lr, decay = 0.99, momentum=0.9).minimize(cost)

#y_op = Y

#predict_op = tf.argmax(Y_pred, 1)

predict_op = tf.argmax(Y, 1)

init = tf.initialize_all_variables()



cost_list = []

err_list = []



with tf.Session() as session:

    session.run(init)

    

    for i in range(max_iter):

        session.run(train_op, feed_dict={X: X_train, T: Y_train_ind})

        if i % print_period == 0:

            test_cost = session.run(cost, feed_dict={X: X_train, T: Y_train_ind})

            Y_prediction_train = session.run(predict_op, feed_dict={X: X_train})

            err_train = error_rate(Y_train, Y_prediction_train)

            Y_prediction = session.run(predict_op, feed_dict={X: X_valid})

            err = error_rate(Y_valid, Y_prediction)

            cost_list.append(test_cost)

            err_list.append(err)

            #prediction = session.run(predict_op, feed_dict={X: X_test})

            print ("i : ", i, "Test cost : ", test_cost, "Error train: ", err_train, "Error : ", err)

        

    test_cost = session.run(cost, feed_dict={X: X_train, T: Y_train_ind})

    #y_test = session.run(y_op, feed_dict={X: X_train})

    Y_prediction = session.run(predict_op, feed_dict={X: X_train})

    err = error_rate(Y_train, Y_prediction)

    score = 1 - err

    #print(y_test)

    #print(Y_prediction)

    Y_test_pred = session.run(predict_op, feed_dict={X: X_test})

    print("Train error : ", err, "Score : ", score)

#print(Y_train_ind)        

#print(prediction)

        
#print(cost_list)

#print(err_list)

print (Y_test_pred)

#print (X_test)

#print (X_train)
# Submission

submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": Y_test_pred

    })

submission.to_csv('titanic4.csv', index=False)
# Plot some column

'''

# Age

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')



# Get average, std, and numer of NaN values in titanic_df

average_age_titanic = titanic_df["Age"].mean()

std_age_titanic = titanic_df["Age"].std()

count_nan_age_titanic = titanic_df["Age"].isnull().sum()



# Generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, 

               average_age_titanic + std_age_titanic, size = count_nan_age_titanic)



# Plot original Age values

# NOTE: drop all null values,and conver to int

titanic_df["Age"].dropna().astype(int).hist(bins=70, ax=axis1)



# Fill NaN values in Age column with random values generated

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1



# Convert from float to int

titanic_df["Age"] = titanic_df["Age"].astype(int)



# Plot New Age values

titanic_df["Age"].hist(bins=70, ax=axis2)

'''

'''

# Plot Age column



# Peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(titanic_df, hue="Survived", aspect=4)

facet.map(sns.kdeplot, "Age", shade=True)

facet.set(xlim=(0, titanic_df["Age"].max()))

facet.add_legend()



# Average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = titanic_df[["Age", "Survived"]].groupby(["Age"],as_index=False).mean()

sns.barplot(x="Age", y="Survived", data=average_age)

'''