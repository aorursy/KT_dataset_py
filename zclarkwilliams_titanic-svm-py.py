# Import the appropriate libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib.learn import SVM as svm
# Generate the test and train data_sets
df_test = pd.read_csv("../input/test.csv")
df_train = pd.read_csv("../input/train.csv")
df_gender = pd.read_csv("../input/gender_submission.csv")
print(df_train)
# Find out which columns are missing data and how much
def get_null_col(data):
    col_null = data.isnull().sum()
    col_names = data.columns[data.isna().any()].tolist()
    return col_names, col_null

# Create table for missing data analysis
def get_null_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = ((data.isnull().sum()/data.isnull().count()).sort_values(ascending=False))*100
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data, percent

def drop_columns(df):
    data = df
    j = 0
    while j < len(data.columns)+1:
        #print(str(df.columns[j]))
        if str(df.columns[j]) == 'Name':
            df.drop('Name', axis=1, inplace=True)
            print("Dropped feature column: %s" % df.columns[j])
        elif df.columns[j] == 'Ticket':
            df.drop('Ticket', axis=1, inplace=True)
            print("Dropped feature column: %s" % df.columns[j])
        elif df.columns[j] == 'Survived':
            df.drop('Survived', axis=1, inplace=True)
            print("Dropped feature column: %s" % df.columns[j])
        elif df.columns[j] == 'Cabin':
            df.drop('Cabin', axis=1, inplace=True)
            print("Dropped feature column: %s" % df.columns[j])
        j += 1
        
    # Get the amount of missing data and drop columns that are missing more than allowable
    missing_data, percentage = get_null_data(df)
    i=0
    for i in range(len(percentage)):
        if percentage[i] > 30:
            bad_column = missing_data[0:i+1].T
            df.drop(str(bad_column.columns.values[0]), axis=1, inplace=True)
    return(df, missing_data)

def encode_data(df):
    # Drop the data that will not help the svm classifications
    df, missing_data = drop_columns(df)
    # Convert the NaN data cells to -1 so svm won't complain
    Rep_Value_NaN = 0
    missing_trans = missing_data.T
    features_missing = missing_trans.columns
    for i in range(len(missing_data['Total'])):
        if missing_data['Total'][i] > 0:
            for i in range(len(df[features_missing[i]])):
                df[features_missing[i]] = df[features_missing[i]].fillna(Rep_Value_NaN)

    # Encode the gender from male and female to 1 and 0
    gender_map = {'male': 0, 'female': 1}
    df.applymap(lambda s: gender_map.get(s) if s in gender_map else s)
    #df['Sex'] = df['Sex'].replace(['male','female'],[0,1])
    # Encode the identified embarked passangers from normal char/string to int
    embark_map = {'S':0, 'Q':1, 'C':2}
    df.applymap(lambda s: embark_map.get(s) if s in embark_map else s)
    #df['Embarked'] = df['Embarked'].replace(['S','Q','C'],[0,1,2])
    return(df)

def input_fn(df):
    data = df
    train_features = encode_data(data)
    return train_features   
df_train['Family_Size'] = df_train['SibSp'] + df_train['Parch']

# Create scatter plot of data
x = df_train['Age']
y = df_train['Survived']
plt.scatter(x=x, y=y)
x = np.asarray(df_train['Age'])
y = np.asarray(df_train['Pclass'])

'''
# Encode gender data to Male=>0 & Female=>1
sex = []
sex_list = np.asarray(df_train['Sex'])
for i in range(len(sex_list)):
    if sex_list[i] == 'male':
        sex.append(0)
    else:
        sex.append(1)
z = sex
'''
z = df_train['Survived']

Ax3D = Axes3D(plt.gcf())
Ax3D.scatter(xs=x, ys=y, zs=z)
# Tweaking display region and labels
Ax3D.set_xlim(0, 80)
Ax3D.set_ylim(0, 3)
Ax3D.set_zlim(0, 1)
Ax3D.set_xlabel('Age')
Ax3D.set_ylabel('Class')
Ax3D.set_zlabel('Suvived')
# Get ID's and survived data 
train_Pids = df_train['PassengerId']
train_Survived = df_train['Survived']
feature = input_fn(df_train)

x_train = feature.astype(int)
y_train = train_Survived.astype(int)
# Extract the features from the training data
feats = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
feature = input_fn(df_test)

x_test = feature.astype(int)
y_test = tdf_train['Survived'].astype(int)
# Building a 3-layer DNN with 50 units each.
classifier_tf = tf.contrib.learn.DNNClassifier(feature_columns=feats, 
                                               hidden_units=[50, 50, 50], 
                                               n_classes=3)
# Use the train data to train this classifier
classifier_tf.fit(x_train, y_train, steps=5000)
# Use the trained model to predict on the test data
predictions = list(classifier_tf.predict(x_test, as_iterable=True))
score = metrics.accuracy_score(y_test, predictions)
def input_func(feature, train_Survived):
    train_inputs = tf.estimator.inputs.numpy_input_fn(
        x = {
            'PassengerId': tf.constant(feature['PassengerId']),
            'Pclass': tf.constant(feature['Pclass']),
            'Sex': tf.constant(feature['Sex']),
            'Sibsp': tf.constant(feature['SibSp']),
            'Parch': tf.constant(feature['Parch']),
            'Fare': tf.constant(feature['Fare']),
            'Age': tf.constant(feature['Age']),
            'Embarked': tf.constant(feature['Embarked']),
            },
        y = tf.constant(train_Survived),
        num_epochs=None,
        shuffle=True)
    return train_inputs

#PassengerId = tf.contrib.layers.real_valued_column(features['PassengerId'])
Pclass = tf.contrib.layers.real_valued_column(feature['Pclass'], dimension=len(feature))
Sex = tf.contrib.layers.real_valued_column(feature['Sex'], dimension=len(feature))
Sibsp = tf.contrib.layers.real_valued_column(feature['SibSp'], dimension=len(feature))
Parch = tf.contrib.layers.real_valued_column(feature['Parch'], dimension=len(feature))
#Ticket = tf.contrib.layers.real_valued_column(feature['Ticket'])
Fare = tf.contrib.layers.real_valued_column(feature['Fare'], dimension=len(feature))
Age = tf.contrib.layers.real_valued_column(feature['Age'], dimension=len(feature))
Embarked = tf.contrib.layers.real_valued_column(feature['Embarked'], dimension=len(feature))
#sq_footage_bucket = tf.contrib.layers.bucketized_column(
#        tf.contrib.layers.real_valued_column('sq_footage'),
#        boundaries=[650.0, 800.0])
#country = tf.contrib.layers.sparse_column_with_hash_bucket(
#        'country', hash_bucket_size=5)
#sq_footage_country = tf.contrib.layers.crossed_column(
#        [sq_footage_bucket, country], hash_bucket_size=10)

svm_classifier = tf.contrib.learn.SVM(
        feature_columns=[Pclass, Sex],#, Sibsp, Parch, Fare, Age, Embarked],
        example_id_column=train_Pids,
        #weight_column_name='weights',
        l1_regularization=0.1,
        l2_regularization=1.0)

svm_classifier.fit(input_fn=input_func(feature, train_Survived), steps=10)
accuracy = svm_classifier.evaluate(input_fn=input_func(feature, train_Survived), steps=1)['accuracy']
