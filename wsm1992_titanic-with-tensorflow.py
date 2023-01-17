import math



from IPython import display

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from sklearn import metrics

import tensorflow as tf

from tensorflow.python.data import Dataset

import datetime

import random

import seaborn as sns

from functools import reduce



tf.logging.set_verbosity(tf.logging.ERROR)

pd.options.display.max_rows = 10

pd.options.display.float_format = '{:.2f}'.format

train = pd.read_csv("../input/train.csv", sep=",")

test = pd.read_csv("../input/test.csv", sep=",")

full = train.append(test)



#train = train.reindex(np.random.permutation(train.index))

full.info()
train.head(5)
train.describe()
f,ax = plt.subplots(2,3,figsize=(16,10))

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[0,0])

sns.countplot('Sex',hue='Survived',data=train,ax=ax[0,1])

sns.countplot('Embarked',hue='Survived',data=train,ax=ax[0,2])

sns.countplot('SibSp',hue='Survived',data=train,ax=ax[1,0])

sns.countplot('Parch',hue='Survived',data=train,ax=ax[1,1])
x = 'Embarked'

y = 'Fare'

hue = 'Pclass'

data = full.copy()

data['Embarked'].fillna('X', inplace=True)

f, ax = plt.subplots(figsize=(8, 5))

fig = sns.boxplot(x=x, y=y,  data=data)

fig.axis(ymin=0, ymax=200);
f, ax = plt.subplots(figsize=(8, 5))

fig = sns.boxplot(x=x, y=y, hue=hue, data=data)

fig.axis(ymin=0, ymax=250);
full['Embarked'].fillna('C', inplace=True)
f,ax = plt.subplots(1,3,figsize=(16,5))



x = 'Embarked'

y = 'Fare'

hue = 'Sex'

data = full.copy()

fig = sns.boxplot(x=x, y=y, hue=hue, data=data,ax=ax[0])

fig.axis(ymin=0, ymax=300);



x = 'Sex'

y = 'Fare'

hue = 'Pclass'

data = full.copy()

fig = sns.boxplot(x=x, y=y, hue=hue, data=data,ax=ax[1])

fig.axis(ymin=0, ymax=300);



x = 'Pclass'

y = 'Fare'

hue = 'Embarked'

data = full.copy()

fig = sns.boxplot(x=x, y=y, hue=hue, data=data,ax=ax[2])

fig.axis(ymin=0, ymax=300);
for sex in full.Sex.unique():

    for pclass in full.Pclass.unique():

        for embarked in full.Embarked.unique():

            features = (full.Sex == sex) & (full.Pclass == pclass) & (full.Embarked == embarked)

            select_nan = np.isnan(full["Fare"]) & features

            full.loc[select_nan,'Fare'] = full[features].Fare.median()
f,ax = plt.subplots(1,5,figsize=(20,5))



x = 'Embarked'

y = 'Age'

data = full.copy()

fig = sns.boxplot(x=x, y=y, data=data,ax=ax[0])

fig.axis(ymin=0, ymax=85);



x = 'Sex'

y = 'Age'

data = full.copy()

fig = sns.boxplot(x=x, y=y,  data=data,ax=ax[1])

fig.axis(ymin=0, ymax=85);



x = 'Pclass'

y = 'Age'

data = full.copy()

fig = sns.boxplot(x=x, y=y,  data=data,ax=ax[2])

fig.axis(ymin=0, ymax=85);



x = 'Parch'

y = 'Age'

data = full.copy()

fig = sns.boxplot(x=x, y=y,  data=data,ax=ax[3])

fig.axis(ymin=0, ymax=85);



x = 'SibSp'

y = 'Age'

data = full.copy()

fig = sns.boxplot(x=x, y=y,  data=data,ax=ax[4])

fig.axis(ymin=0, ymax=85);
for sibSp in full.SibSp.unique():

    for pclass in full.Pclass.unique():

        for embarked in full.Embarked.unique():

            features = (full.SibSp == sibSp) & (full.Pclass == pclass) & (full.Embarked == embarked)

            select_nan = np.isnan(full["Age"]) & features

            full.loc[select_nan,'Age'] = full[features].Age.mean()
full.info()
full['Age'].fillna(full[(full.SibSp == 2) & (full.Pclass == 3)]['Age'].mean(), inplace=True)

full.info()
a = sns.FacetGrid( train, hue = 'Survived', aspect=3 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , train['Age'].max()))

a.add_legend()



a = sns.FacetGrid( train, hue = 'Survived', aspect=3 )

a.map(sns.kdeplot, 'Fare', shade= True )

a.set(xlim=(0 , train['Fare'].quantile(0.95)))

a.add_legend()
age_boundaries = [14, 30, 40, 49, 57]

fare_boundaries = [18, 25]
train.info()
full['FamilySize'] = full.SibSp + full.Parch + 1

train = full.head(891)

test = full.tail(418)

sns.countplot('FamilySize',hue='Survived',data=train)
family_size_boundaries=[1, 4]
full['Single'] = full.FamilySize.apply(lambda fs: True if fs == 1 else False)
full.Name.sample(10)
full['Title'] = full['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

full['Title'].unique()
train = full.head(891)

test = full.tail(418)



title_names = (train['Title'].value_counts() > 2) 

train.insert(loc = len(train.columns),column='BigTitle', value=train['Title'].apply(lambda x: title_names.loc[x]))

train[train.BigTitle == True].Title.unique()

f,ax = plt.subplots(1,2,figsize=(16,5))

sns.countplot('Title',hue='Survived',data=train[train.BigTitle == True], ax=ax[0])

sns.countplot('Title',hue='Survived',data=train[train.BigTitle == False], ax=ax[1])
full['Title'] = full['Title'].apply(lambda title: 'Don' if not title_names.index.contains(title) else title)

full['Title'] = full['Title'].apply(lambda title: title if title_names.loc[title] == True else 'X')

full.Title.unique()
full['NameLength'] = full['Name'].apply(lambda name: len(name))

train = full.head(891)

test = full.tail(418)

a = sns.FacetGrid( train, hue = 'Survived', aspect=3 )

a.map(sns.kdeplot, 'NameLength', shade= True )

a.set(xlim=(0 , train['NameLength'].quantile(0.95)))

a.add_legend()
name_length_boundaries = [12, 28]
full.columns
a = sns.FacetGrid( train,col='Sex', hue = 'Survived', aspect=3 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , train['Age'].max()))

a.add_legend()



a = sns.FacetGrid( train,col='Pclass', hue = 'Survived', aspect=3 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , train['Age'].max()))

a.add_legend()



a = sns.FacetGrid( train,col='Pclass', hue = 'Survived', aspect=3 )

a.map(sns.kdeplot, 'Fare', shade= True )

a.set(xlim=(0 , 100))

a.add_legend()



a = sns.FacetGrid( train,col='Sex', hue = 'Survived', aspect=3 )

a.map(sns.kdeplot, 'NameLength', shade= True )

a.set(xlim=(0 , 100))

a.add_legend()
a = sns.FacetGrid( train,col='Sex', row='Single', hue = 'Survived', aspect=3 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , 100))

a.add_legend()
sex_cross_age_boundaries = [15, 26, 32,46, 54]

parch_cross_age_boundaries = [5, 10, 18, 30, 35]

pclass_cross_age_boundaries = [18, 30, 36, 40, 47]

pclass_cross_fare_boundaries = [8,18,25, 55]

sex_cross_name_length_boundaries = [12, 26, 42]
train = full.head(891)

test = full.tail(418)



train['SexCode'] = train.Sex.apply(lambda sex: 1 if sex == 'male' else 0)



f,ax = plt.subplots(2,2,figsize=(20,16))



sns.swarmplot(x='Single',y='Pclass',hue='Survived',data=train,palette='husl',ax=ax[0,0])

sns.swarmplot(x='Parch',y='SexCode',hue='Survived',data=train,palette='husl',ax=ax[0,1])

sns.swarmplot(x='Embarked',y='Pclass',hue='Survived',data=train,palette='husl',ax=ax[1,0])

sns.swarmplot(x='Embarked',y='SexCode',hue='Survived',data=train,palette='husl',ax=ax[1,1])
full['Parch'] = full['Parch'].apply(lambda parch: parch if parch <= 2 else 2)

full['SibSp'] = full['SibSp'].apply(lambda parch: parch if parch <= 5 else 5)
train = full.head(891)

test = full.tail(418)

train['SexCode'] = train.Sex.apply(lambda sex: 1 if sex == 'male' else 0)

#train = pd.get_dummies(data=train, columns = ['Sex'])

corrmat = train[['Survived', 'SexCode', 'Pclass', 'Age', 'SibSp', 'FamilySize', 'Parch', 'Fare']].corr()

f, ax = plt.subplots(figsize=(12, 9))

colormap = plt.cm.RdBu

sns.heatmap(corrmat,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    """Trains a linear regression model of one feature.



    Args:

      features: pandas DataFrame of features

      targets: pandas DataFrame of targets

      batch_size: Size of batches to be passed to the model

      shuffle: True or False. Whether to shuffle the data.

      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely

    Returns:

      Tuple of (features, labels) for next data batch

    """

    

    # Convert pandas data into a dict of np arrays.

    features = {key:np.array(value) for key,value in dict(features).items()}                             



    # Construct a dataset, and configure batching/repeating

    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit      

    ds = ds.batch(batch_size).repeat(num_epochs)

    

    # Shuffle the data, if specified

    if shuffle:

        ds = ds.shuffle(10000)



    # Return the next batch of data

    features, labels = ds.make_one_shot_iterator().get_next()



    return features, labels
def train_linear_classifier_model(

    learning_rate,

    steps,

    batch_size,

    periods,

    regularization_strength,

    training_examples,

    training_targets,

    validation_examples,

    validation_targets):

    """Trains a linear regression model of one feature.

  

  In addition to training, this function also prints training progress information,

  as well as a plot of the training and validation loss over time.

  

  Args:

    learning_rate: A `float`, the learning rate.

    steps: A non-zero `int`, the total number of training steps. A training step

      consists of a forward and backward pass using a single batch.

    batch_size: A non-zero `int`, the batch size.

    training_examples: A `DataFrame` containing one or more columns from

      `california_housing_dataframe` to use as input features for training.

    training_targets: A `DataFrame` containing exactly one column from

      `california_housing_dataframe` to use as target for training.

    validation_examples: A `DataFrame` containing one or more columns from

      `california_housing_dataframe` to use as input features for validation.

    validation_targets: A `DataFrame` containing exactly one column from

      `california_housing_dataframe` to use as target for validation.

      

  Returns:

    A `LinearClassifier` object trained on the training data.

  """



    steps_per_period = steps / periods

    

    # Create a linear classifier object.

    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)

    #my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)    

    linear_classifier = tf.estimator.DNNClassifier(

    #linear_classifier = tf.estimator.LinearClassifier(

      feature_columns=construct_feature_columns(training_examples),

      hidden_units=[10, 10],

      optimizer=my_optimizer

    )

    

    # Create input functions

    training_input_fn = lambda: my_input_fn(training_examples, 

                                          training_targets["target"], 

                                          batch_size=batch_size)

    predict_training_input_fn = lambda: my_input_fn(training_examples, 

                                                  training_targets["target"], 

                                                  num_epochs=1, 

                                                  shuffle=False)

    predict_validation_input_fn = lambda: my_input_fn(validation_examples, 

                                                    validation_targets["target"], 

                                                    num_epochs=1, 

                                                    shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess

    # loss metrics.

    print("Training model...")

    print("LogLoss (on training data):")

    training_log_losses = []

    validation_log_losses = []

    for period in range (0, periods):

        # Train the model, starting from the prior state.

        linear_classifier.train(

            input_fn=training_input_fn,

            steps=steps_per_period

        )

        # Take a break and compute predictions.    

        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)

        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])



        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)

        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])



        training_log_loss = metrics.log_loss(training_targets, training_probabilities)

        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)

        # Occasionally print the current loss.

        print( "  period %02d : %0.2f" % (period, training_log_loss))

        # Add the loss metrics from this period to our list.

        training_log_losses.append(training_log_loss)

        validation_log_losses.append(validation_log_loss)

    print("Model training finished.")

    

    # Output a graph of loss metrics over periods.

    plt.ylabel("LogLoss")

    plt.xlabel("Periods")

    plt.title("LogLoss vs. Periods")

    plt.tight_layout()

    plt.plot(training_log_losses, label="training")

    plt.plot(validation_log_losses, label="validation")

    plt.legend()



    return linear_classifier
def preprocess_features(df):

    """Prepares input features from tantic data set.



    Args:

    df: A Pandas DataFrame expected to contain data

      from the train data set.

    Returns:

    A DataFrame that contains the features to be used for the model, including

    synthetic features.

    """

    selected_features = df[

        ['Sex', 'Pclass', 'Age', 'Parch', 'SibSp', 'FamilySize', 'Single', 'Fare', 'Title', 'Embarked', 'NameLength']]

    processed_features = selected_features.copy()

    

    return processed_features



def preprocess_targets(df):

    """Prepares target features (i.e., labels) from tantic data set.



    Args:

    df: A Pandas DataFrame expected to contain data

      from the train data set.

    Returns:

    A DataFrame that contains the target feature.

    """

    output_targets = pd.DataFrame()

    output_targets["target"] =  df['Survived'] 

    return output_targets
train = full.head(891)

test = full.tail(418)



training_examples = preprocess_features(train.head(700))

training_targets = preprocess_targets(train.head(700))



validation_examples = preprocess_features(train.tail(291))

validation_targets = preprocess_targets(train.tail(291))



# Double-check that we've done the right thing.

print ("Training examples summary:")

display.display(training_examples.describe())

print( "Validation examples summary:")

display.display(validation_examples.describe())



#print( "Training targets summary:")

#display.display(training_targets.describe())

#print( "Validation targets summary:")

#display.display(validation_targets.describe())
def cross_columns(crolss_array, hash_bucket_size=1000):

    cross_column = tf.feature_column.indicator_column(tf.feature_column.crossed_column(crolss_array , hash_bucket_size=hash_bucket_size))

    return cross_column

    

def construct_feature_columns(input_features):

    """Construct the TensorFlow Feature Columns.



    Args:

    input_features: The names of the numerical input features to use.

    Returns:

    A set of feature columns

    """

    features = []



    sex_categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key='Sex',vocabulary_list=["M", "F"])

    sex_indicator_column = tf.feature_column.indicator_column(sex_categorical_column)

    features.append(sex_indicator_column)



    pclass_categorical_column = tf.feature_column.categorical_column_with_identity(key='Pclass',num_buckets=4)

    pclass_indicator_column = tf.feature_column.indicator_column(pclass_categorical_column)

    features.append(pclass_indicator_column)

    

    embarked_categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key='Embarked',vocabulary_list=["S", "C", "Q"])

    embarked_indicator_column = tf.feature_column.indicator_column(embarked_categorical_column)

    features.append(embarked_indicator_column)

        

    title_categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key='Title',vocabulary_list=full.Title.unique())

    title_indicator_column = tf.feature_column.indicator_column(title_categorical_column)

    features.append(title_indicator_column)

    

    name_length_categorical_column = tf.feature_column.numeric_column("NameLength")

    name_length_bucket_column = tf.feature_column.bucketized_column(name_length_categorical_column, boundaries=name_length_boundaries)

    features.append(name_length_bucket_column)



    parch_categorical_column = tf.feature_column.categorical_column_with_identity(key='Parch',num_buckets=4)

    parch_indicator_column = tf.feature_column.indicator_column(parch_categorical_column)

    #features.append(parch_indicator_column)

    

    sibsp_categorical_column = tf.feature_column.categorical_column_with_identity(key='SibSp',num_buckets=4)

    sibsp_indicator_column = tf.feature_column.indicator_column(sibsp_categorical_column)

    #features.append(sibsp_indicator_column)

    

    family_size_categorical_column = tf.feature_column.numeric_column("FamilySize")

    family_size_bucket_column = tf.feature_column.bucketized_column(family_size_categorical_column, boundaries=family_size_boundaries)

    features.append(family_size_bucket_column)

    

    single_numric_column = tf.feature_column.numeric_column('Single')

    features.append(single_numric_column)

    

    age_categorical_column = tf.feature_column.numeric_column("Age")

    age_bucket_column = tf.feature_column.bucketized_column(age_categorical_column, boundaries=age_boundaries)

    #features.append(age_bucket_column)

    

    fare_categorical_column = tf.feature_column.numeric_column("Fare")

    fare_bucket_column = tf.feature_column.bucketized_column(fare_categorical_column, boundaries=fare_boundaries)

    features.append(fare_bucket_column)    

    

    

    sex_cross_age_bucket_column = tf.feature_column.bucketized_column(age_categorical_column, boundaries=age_boundaries)

    features.append(cross_columns(['Sex', 'Single', sex_cross_age_bucket_column]))        

            

    parch_cross_age_bucket_column = tf.feature_column.bucketized_column(age_categorical_column, boundaries=parch_cross_age_boundaries)

    #features.append(cross_columns(['Parch', parch_cross_age_bucket_column]))

    

    pclass_cross_fare_bucket_column = tf.feature_column.bucketized_column(fare_categorical_column, boundaries=pclass_cross_fare_boundaries)

    #features.append(cross_columns(['Pclass', pclass_cross_fare_bucket_column]))

    

    pclass_cross_age_bucket_column = tf.feature_column.bucketized_column(age_categorical_column, boundaries=pclass_cross_age_boundaries)

    #features.append(cross_columns(['Pclass', pclass_cross_age_bucket_column]))

    

    sex_cross_name_length_bucket_column = tf.feature_column.bucketized_column(name_length_categorical_column, boundaries=sex_cross_name_length_boundaries)

    #features.append(cross_columns(['Sex', sex_cross_name_length_bucket_column]))

        

    #features.append(cross_columns(['Sex', 'Pclass']))

    features.append(cross_columns(['SibSp', 'Parch'], 18))

    features.append(cross_columns(['SibSp', 'Sex'], 12))

    features.append(cross_columns(['Single', 'Pclass'], 6))

    

    #features.append(cross_columns([age_bucket_column, 'Pclass']))

    

    #features.append(cross_columns([age_bucket_column, 'Sex']))

    

    #features.append(cross_columns(['Embarked', 'Sex']))

    

    #features.append(cross_columns(['Embarked', age_bucket_column]))

    

    features.append(cross_columns(['Embarked', 'Pclass']))



    feature_columns = set(features)

    return feature_columns
linear_classifier = train_linear_classifier_model(

    learning_rate=0.16,

    steps=200,

    batch_size=500,

    periods=15,

    regularization_strength=0.015,

    training_examples=training_examples,

    training_targets=training_targets,

    validation_examples=validation_examples,

    validation_targets=validation_targets)
predict_validation_input_fn = lambda: my_input_fn(validation_examples, 

                                                    validation_targets["target"], 

                                                    num_epochs=1, 

                                                    shuffle=False)



evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print(evaluation_metrics.keys())

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])

print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])

print(evaluation_metrics)
validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)

# Get just the probabilities for the positive class

validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])



false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(

    validation_targets, validation_probabilities)

plt.plot(false_positive_rate, true_positive_rate, label="our model")

plt.plot([0, 1], [0, 1], label="random classifier")

_ = plt.legend(loc=2)



true_positive_rate
def assign_probability(df, linear_classifier, validation=False,field='probability'):

    result = df.copy()

    fake = df.copy()

    fake['Survived'] = 0

    v_examples = preprocess_features(result)

    if validation:

        v_targets = preprocess_targets(result)

    else:

        v_targets = preprocess_targets(fake)

    predict_validation_input_fn = lambda: my_input_fn(v_examples, 

                                                        v_targets['target'], 

                                                        num_epochs=1, 

                                                        shuffle=False)

    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)

    result[field] = np.array([item['probabilities'][1] for item in validation_probabilities])

    return result
result = assign_probability(test, linear_classifier,False)

validation = assign_probability(train, linear_classifier,True)

validation
def find_treshold(validation):

    best_accuracy = 0

    best_threshold = 0

    target = 'Survived'

    for i in range(0, 101):

        threshold = i/100.0

        validation['new_survived'] = validation['probability'].apply(lambda p: 1 if p >= threshold else 0)

        accuracy = validation[validation['new_survived'] == validation['Survived']]['Survived'].count()/validation['Survived'].count().astype(float)

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_threshold = threshold

    threshold = best_threshold

    validation['new_survived'] = validation['probability'].apply(lambda p: 1 if p >= best_threshold else 0)



    p = validation[validation['probability'] >= threshold]

    n = validation[validation['probability'] < threshold]

    tp = p[p[target] == 1]

    fp = p[p[target] == 0]

    tn = n[n[target] == 0]

    fn = n[n[target] == 1]



    pn = p['Survived'].count().astype(float)

    nn = n['Survived'].count().astype(float)

    tpn = tp['Survived'].count().astype(float)

    fpn = fp['Survived'].count().astype(float)

    tnn = tn['Survived'].count().astype(float)

    fnn = fn['Survived'].count().astype(float)



    print ('best_threshold: %s' % threshold)

    print ('best_accuracy: %s' % best_accuracy)

    print ('result number: %s' % pn)

    print ('tpn: %s' % tpn)

    print ('fpn: %s' % fpn)

    print ('tnn: %s' % tnn)

    print ('fnn: %s' % fnn)



    precision = tpn / pn

    tp_rate = tpn / (tpn + fnn)

    fp_rate = fpn / (fpn + tnn)

    precision_n = tnn / (tnn + fnn)



    print ('precision: %s' % precision)

    print ('tp_rate: %s' % tp_rate)

    print ('fp_rate: %s' % fp_rate)

    print ('precision_n: %s' % precision_n)

    return best_threshold
threshold = find_treshold(validation)

threshold
result["Survived"] = result["probability"].apply(lambda a: 1 if a > threshold else 0)

evaluation = result[["PassengerId", "Survived"]]

evaluation
evaluation.to_csv("evaluation_submission.csv",index=False)