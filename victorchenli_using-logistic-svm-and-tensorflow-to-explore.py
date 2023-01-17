import tensorflow as tf
# pandas

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")

test_validation = pd.read_csv("../input/gendermodel.csv")
test_df = test_df.join(test_validation.set_index("PassengerId"),on="PassengerId", how="left")
test_df.info()
all_df = pd.concat([train_df,test_df], axis=0)
all_df.info()
#convert title and lastname to integer

def split_name(fullName):

    firstName, lastName = fullName.split(",")

    titleIndex = lastName.find(".")

    title, lastName = lastName.split(".", 1)

    return [title,lastName]



def get_title(name):

    title = split_name(name['Name'])[0]

    return hash(title)%256

    

def get_lastName(name):

    lastName = split_name(name['Name'])[1]

    return hash(lastName)%1024



#convert sex to 1,0

def conv_sex(row):

    if row['Sex']=='male':

        return 1

    else:

        return 0
type(all_df['Name'])
all_df['Name'] = all_df['Name'].astype(str)

all_df['Title'] = all_df[['Name']].apply(get_title,axis=1)

all_df['LastName'] = all_df[['Name']].apply(get_lastName,axis=1)

all_df = all_df.drop(['Name'], axis=1)
all_df['Sex'] = all_df['Sex'].astype(str)

all_df['Sex'] = all_df.apply(conv_sex, axis=1)
# get average, std, and number of NaN values in titanic_df

average_age_titanic   = all_df["Age"].mean()

std_age_titanic       = all_df["Age"].std()

count_nan_age_titanic = all_df["Age"].isnull().sum()
rand = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

all_df["Age"][np.isnan(all_df["Age"])] = rand
#extract ticket class info from ticket field

def get_ticketClass(row):

    ticketInfo = row["Ticket"].split(" ",1)

    if len(ticketInfo) > 1: 

        return hash(ticketInfo[0])%256+1

    else:

        return 0
all_df['TicketClass'] = all_df[['Ticket']].apply(get_ticketClass, axis=1)
all_df = all_df.drop(['Ticket'],axis=1)
#Extract Cabin number

def get_cabinNum(row):

    if pd.isnull(row['Cabin']):

        return 0

    else:

        cabinInfo = row["Cabin"].split(" ")

        return len(cabinInfo)
all_df['CabinNum'] = all_df[['Cabin']].apply(get_cabinNum, axis=1)

all_df = all_df.drop(['Cabin'],axis=1)
#Extract Cabin number

def get_embarked(row):

    return hash(row["Embarked"])%256
all_df['Embarked'] = all_df[['Embarked']].apply(get_embarked, axis=1)
#Fare

average_fare_titanic   = test_df["Fare"].mean()

std_fare_titanic       = test_df["Fare"].std()

count_nan_fare_titanic = test_df["Fare"].isnull().sum()
rand_fare = np.random.randint(average_fare_titanic - std_fare_titanic, average_fare_titanic + std_fare_titanic, size = count_nan_fare_titanic)

all_df["Fare"][np.isnan(all_df["Fare"])] = rand_fare

all_features = all_df.drop("Survived",axis=1)
all_outputs = all_df['Survived']
# Normalize the inputs so they have ~0 mean, and 1 Standard Deviation

# make trainning easiler

all_features = (all_features - all_features.mean(axis=0)) / all_features.std(axis=0)
all_df.drop("PassengerId",axis=1,inplace=True)
from sklearn.model_selection import train_test_split
X = all_features

y = all_outputs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# start with logistic ml

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
# print report

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
# start with SVM

from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)
# add Grid Search to find best parama

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train,y_train)
# print report

from sklearn.metrics import classification_report,confusion_matrix

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))

print(classification_report(y_test,predictions))
# start with tensorflow version 1



input_vector_size = X_train.shape[1]

output_size = 1

number_of_hidden_neurons = 5



y_train = np.expand_dims(y_train, 1)

y_test = np.expand_dims(y_test, 1)



# === We build the graph here!

titanic_graph_1 = tf.Graph()



with titanic_graph_1.as_default():

    

    # We create "None" size placeholders to let us put variable sized "Batches" of data at a time

    x = tf.placeholder("float", shape = [None, input_vector_size])

    y = tf.placeholder("float", shape =[None, output_size])



    # We're going to use an initializer to generate random values for our weights

    initializer = tf.contrib.layers.xavier_initializer()



    # Hidden layer weights, connecting input to hidden neurons

    hidden_weights = tf.Variable(initializer(shape=[input_vector_size, number_of_hidden_neurons]))

    

    # Output layer weights, connecting hidden neurons to output

    output_weights = tf.Variable(initializer(shape=[number_of_hidden_neurons, output_size]))

    # Biases for the hidden neurons

    bias = tf.Variable(tf.zeros([output_size]))

    

    # Biases for the output 

    bias1 = tf.Variable(tf.zeros([number_of_hidden_neurons]))

    

    # Hidden layer logits and activation

    hidden = tf.nn.tanh(tf.matmul(x, hidden_weights) + bias1)

    

    # Output layer 

    output_layer = (tf.matmul(hidden, output_weights) + bias)

    

    # Squared Error function

    error = tf.reduce_mean(tf.pow((y-output_layer), 2))

    

    # We will use Adam Optimizer for network optimization

    optimizer = tf.train.AdamOptimizer().minimize(error)

    

    # Our initialization operation

    init = tf.global_variables_initializer()
# We create our sessions

sess_1 = tf.Session(graph=titanic_graph_1)



# Make sure to run the initialization

sess_1.run(init)



Total_ephoch = 20000



train_error = []

valid_error = []



# Train loop for the model

for i in range(Total_ephoch):

    

    #Session runs train_op to minimize loss

    sess_1.run(optimizer, feed_dict={x: X_train, y: y_train})

    

    train_error.append(sess_1.run(error, feed_dict={x: X_train, y: y_train}))

    valid_error.append(sess_1.run(error, feed_dict={x: X_test, y: y_test}))

    

    if i%1000 == 0:

        print ("validation error:", valid_error[i])
