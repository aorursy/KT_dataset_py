# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # for plotting graphs
import seaborn # statistical data visualization library
from functools import cmp_to_key

from sklearn import metrics
import math
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input/"))
print(os.listdir("./"))

# Any results you write to the current directory are saved as output.
# load the train.csv file
df = pd.read_csv('../input/train.csv')
df = df.reindex()
df.describe()
df_test = pd.read_csv('../input/test.csv')
df_test.describe()
print('total fields = %d' % (len(df.keys())))
print('fields = %s' % (df.keys()))
def min2(l, default = 0.0):
    if len(l) == 0:
        return default
    else:
        return min(l)

def max2(l, default = 0.0):
    if len(l) == 0:
        return default
    else:
        return max(l)

def avg2(l, default = 0.0):
    if len(l) == 0:
        return default
    else:
        return float(sum(l)) / float(len(l))

def std2(l, default = 0.0):
    if len(l) == 0:
        return default
    else:
        return np.std(l)

def histogram_for_non_numerical_series(s):
    d = {}
    for v in s:
        d[v] = d.get(v, 0) + 1
    bin_s_label = list(d.keys())
    bin_s_label.sort()
    bin_s = list(range(0, len(bin_s_label)))
    hist_s = [d[v] for v in bin_s_label]
    bin_s.append(len(bin_s))
    bin_s_label.insert(0, '_')
    return (hist_s, bin_s, bin_s_label)
    
def plot_hist_with_target3(plt, df, feature, target, histogram_bins = 10):
    # reference:
    #    https://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
    #    https://matplotlib.org/gallery/api/two_scales.html 
    #    https://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html
    #    https://matplotlib.org/2.0.0/examples/color/named_colors.html
    #    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xticks.html
    title = feature
    plt.title(title)
    s = df[feature]
    t = df[target]
    t_max = max(t)
    # get histogram of the feature
    bin_s_label = None
    # fillna with 0.0 or '_N/A_'
    na_cnt = sum(s.isna())
    if na_cnt > 0:
        if True in [type(_) == str for _ in s]:
            print('found %d na in string field %s' % (na_cnt, feature))
            s = s.fillna('_N/A_')
        else:
            print('found %d na in numerical field %s' % (na_cnt, feature))
            s = s.fillna(-1.0)
    try:
        hist_s, bin_s = np.histogram(s, bins = histogram_bins)
    except Exception as e:
        # print('ERROR: failed to draw histogram for %s: %s: %s' % (name, type(e).__name__, str(e)))
        hist_s, bin_s, bin_s_label = histogram_for_non_numerical_series(s)
        # return
    # histogram of target by distribution of feature
    hist_t_by_s_cnt = [0] * (len(bin_s) - 1)
    hist_t_by_s = [] 
    for i in range(0, (len(bin_s) - 1)):
        hist_t_by_s.append([])
    # get target histogram for numerical feature
    if bin_s_label is None:
        for (sv, tv) in zip(s, t):
            pos = 0
            for i in range(0, len(bin_s) - 1):
                if sv >= bin_s[i]:
                    pos = i
            hist_t_by_s_cnt[pos] += 1
            hist_t_by_s[pos].append(tv)
    else:
        for (sv, tv) in zip(s, t):
            pos = bin_s_label.index(sv) - 1
            hist_t_by_s_cnt[pos] += 1
            hist_t_by_s[pos].append(tv)
        # count avg, to re-sort bin_s and bin_s_label by avg
        hist_t_by_s_avg = [float(avg2(n)) for n in hist_t_by_s]
        # hist_t_by_s_std = [float(std2(n)) for n in hist_t_by_s]
        # hist_t_by_s_adj = list(np.array(hist_t_by_s_avg) + np.array(hist_t_by_s_std))
        hist_t_by_s_adj = hist_t_by_s_avg
        # print('before sort:\n%s\n%s\n%s' % (bin_s, bin_s_label, hist_t_by_s_adj))
        bin_hist_label = list(zip(bin_s[1:], hist_t_by_s_adj, bin_s_label[1:]))
        bin_hist_label.sort(key = cmp_to_key(lambda x, y: x[1] - y[1]))
        (bin_s, hist_t_by_s_adj, bin_s_label) = zip(*bin_hist_label)
        bin_s = list(bin_s)
        hist_t_by_s_adj = list(hist_t_by_s_adj)
        bin_s_label = list(bin_s_label)
        bin_s.insert(0, 0)
        bin_s_label.insert(0, '_')
        # re-arrange hist_s and hist_t_by_s
        hist_s_new = []
        hist_t_by_s_new = []
        for i in bin_s[1:]:
            hist_s_new.append(hist_s[i - 1])
            hist_t_by_s_new.append(hist_t_by_s[i - 1])
        hist_s = hist_s_new
        hist_t_by_s = hist_t_by_s_new
        # print('after sort:\n%s\n%s\n%s' % (bin_s, bin_s_label, hist_t_by_s_adj))
        # reset bin_s's ordering
        bin_s.sort()
    hist_s = list(hist_s)
    if len(hist_s) < len(bin_s):
        hist_s.insert(0, 0.0)
    hist_s_max = max(hist_s)
    plt.fill_between(bin_s, hist_s, step = 'mid', alpha = 0.5, label = feature)
    if bin_s_label is not None:
        plt.xticks(bin_s, bin_s_label)
    plt.xticks(rotation = 90)
    # just to show legend for ax2
    # plt.errorbar([], [], yerr = [], fmt = 'ok', lw = 3, ecolor = 'sienna', mfc = 'sienna', label = target)
    plt.legend(loc = 'upper right')
    hist_t_by_s = list(hist_t_by_s)
    if len(hist_t_by_s) < len(bin_s):
        hist_t_by_s.insert(0, [0.0])
    hist_t_by_s_min = [float(min2(n)) for n in hist_t_by_s]
    hist_t_by_s_max = [float(max2(n)) for n in hist_t_by_s]
    hist_t_by_s_avg = [float(avg2(n)) for n in hist_t_by_s]
    hist_t_by_s_std = [float(std2(n)) for n in hist_t_by_s]
    hist_t_by_s_err = [np.array(hist_t_by_s_avg) - np.array(hist_t_by_s_min), np.array(hist_t_by_s_max) - np.array(hist_t_by_s_avg)]
    plt.xlabel(feature)
    plt.ylabel('Count')
    ax2 = plt.twinx()
    ax2.grid(False)
    ax2.errorbar(bin_s, hist_t_by_s_avg, yerr = hist_t_by_s_err, fmt='.k', lw = 1, ecolor = 'sienna')
    ax2.errorbar(bin_s, hist_t_by_s_avg, yerr = hist_t_by_s_std, fmt='ok', lw = 3, ecolor = 'sienna', mfc = 'sienna', label = target)
    ax2.set_ylabel(target)
    plt.legend(loc = 'upper left')
    plt.tight_layout()

plt.figure(figsize = (20, 90))
i = 1
for name in df.keys():
    plt.subplot(21, 4, i)
    plot_hist_with_target3(plt, df, name, 'SalePrice', histogram_bins = 'rice')
    i += 1
plt.tight_layout()
df.corr()
plt.figure(figsize = (10, 10))
seaborn.heatmap(df.corr())
numerical_fields = [
    'OverallQual', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea', 'MSSubClass'
]

categorical_fields = [
    'Neighborhood', 'RoofMatl', 'CentralAir', 'Electrical', 'KitchenQual'
]

fields = numerical_fields + categorical_fields

plt.figure(figsize = (20, 90))
i = 1
for name in fields:
    plt.subplot(21, 4, i)
    plot_hist_with_target3(plt, df, name, 'SalePrice', histogram_bins = 'rice')
    i += 1
plt.tight_layout()
def log2(n):
    if n <= 0:
        return 0.0
    else:
        return math.log(n)

def get_logged_series(s):
    s1 = [log2(n) for n in s]
    return s1
    
def get_logged_df(df):
    df1 = pd.DataFrame()
    for k in df.keys():
        s1 = get_logged_series(s)
        df1[k] = s1
    return df1

def get_scaled_series(base = 1.0):
    def wrapper_get_scaled_series(s):
        s_max = max(s)
        return [base * float(n) / float(s_max) for n in s]
    return wrapper_get_scaled_series

def get_scaled_df(df, base = 1.0):
    df1 = pd.DataFrame()
    for k in df.keys():
        s = df[k]
        s1 = get_scaled_series(base = base)(s)
        df1[k] = s1
    return df1

def get_dummies(dummy_na = False):
    def wrapper_get_dummies(s):
        df = pd.get_dummies(s, prefix = s.name, dummy_na = dummy_na)
        df1 = pd.DataFrame()
        for k in df.keys():
            s = df[k]
            name = s.name
            name = name.replace('(', '')
            name = name.replace(')', '')
            name = name.replace(' ', '')
            s1 = s.rename(name)
            df1[s1.name] = s1
        return df1
    return wrapper_get_dummies

def convert_features(df, num_fields, cat_fields, num_fields_proc = None, cat_fields_proc = None, label_name = None, train_validate_ratio = None):
    if num_fields_proc is None:
        num_fields_proc = lambda x: x
    if cat_fields_proc is None:
        cat_fields_proc = lambda x: x
    features = pd.DataFrame()
    for k in num_fields:
        features[k] = num_fields_proc(df[k].copy())
    for k in cat_fields:
        features = features.join(cat_fields_proc(df[k].copy()))
    if label_name is not None:
        labels = df[label_name].copy()
    else:
        labels = None
    if train_validate_ratio is None:
        train_validate_ratio = 1
    train_num = int(len(df) * train_validate_ratio)
    validate_num = len(df) - train_num
    train_features = features.head(train_num)
    validate_features = features.tail(validate_num)
    if labels is not None:
        train_labels = labels.head(train_num)
        validate_labels = labels.tail(validate_num)
    else:
        train_labels = None
        validate_labels = None
    return (train_features, train_labels, validate_features, validate_labels)
def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
    input_features: The names of the numerical input features to use.
    Returns:
    A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.

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
    # features = {key:np.array(value) for key,value in dict(features).items()}                                            

    #
    # note: can convert to dict directly
    #
    features = dict(features)


    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def my_input_fn_pred(features, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.

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
    # features = {key:np.array(value) for key,value in dict(features).items()}                                            

    #
    # note: can convert to dict directly
    #
    features = dict(features)


    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices(features) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features = ds.make_one_shot_iterator().get_next()
    return features

def train_dnn_regressor_model(
    optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    """Trains a DNN regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.

    Returns:
    A `LinearRegressor` object trained on the training data.
    """
    # RMSE, RMSLE: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    if validation_examples is not None and validation_targets is not None:
        do_validation = True
    else:
        do_validation = False
    
    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples, 
                                            training_targets, 
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                    training_targets, 
                                                    num_epochs=1, 
                                                    shuffle=False)
    if do_validation == True:
        predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                          validation_targets, 
                                                          num_epochs=1, 
                                                          shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE, RMSLE (on training data):")
    training_rmse = []
    validation_rmse = []
    training_rmsle = []
    validation_rmsle = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        
        # Compute training loss RMSE.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_targets, training_predictions))

        # Compute training loss RMSLE.
        training_root_mean_squared_log_error = math.sqrt(
            metrics.mean_squared_log_error(training_targets, training_predictions))
        
        if do_validation == True:
            validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
            validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

            # Compute validation loss RMSE.
            validation_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(validation_targets, validation_predictions))

            # Compute validation loss RMSLE.
            validation_root_mean_squared_log_error = math.sqrt(
                metrics.mean_squared_log_error(validation_targets, validation_predictions))

        # Occasionally print the current loss.
        print("  period %02d : %0.2f, %0.4f" % (period, training_root_mean_squared_error, training_root_mean_squared_log_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        training_rmsle.append(training_root_mean_squared_log_error)
        if do_validation == True:
            validation_rmse.append(validation_root_mean_squared_error)
            validation_rmsle.append(validation_root_mean_squared_log_error)
 
    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.figure(figsize = (15, 5))
    # RMSE
    plt.subplot(1, 3, 1)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.plot(training_rmse, label="training")
    if do_validation == True:
        plt.plot(validation_rmse, label="validation")
    plt.legend()
    # RMSLE
    plt.subplot(1, 3, 2)
    plt.ylabel("RMSLE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Logarithmic Error vs. Periods")
    plt.plot(training_rmsle, label="training")
    if do_validation == True:
        plt.plot(validation_rmsle, label="validation")
    plt.legend()
    # Target / Prediction
    plt.subplot(1, 3, 3)
    plt.ylabel("Target")
    plt.xlabel("Prediction")
    plt.title("Target vs. Prediction")
    lim = max(training_targets)
    if do_validation == True:
        lim = max(lim, max(validation_targets))
    lim *= 1.05
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.plot([0, lim], [0, lim], alpha = 0.5, color = 'red')
    plt.scatter(training_predictions, training_targets, alpha = 0.5, label="training")
    if do_validation == True:
        plt.scatter(validation_predictions, validation_targets, alpha = 0.5, label="validation")
    plt.legend()
    plt.tight_layout()
    
    return dnn_regressor

numerical_fields = [
    'OverallQual', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea', 'MSSubClass'
]

categorical_fields = []


ret = convert_features(df, 
                       numerical_fields, 
                       categorical_fields, 
                       num_fields_proc = None, 
                       cat_fields_proc = get_dummies(dummy_na = True),
                       label_name = 'SalePrice',
                       train_validate_ratio = 0.7)


training_features = ret[0]
training_targets = ret[1]
validation_features = ret[2]
validation_targets = ret[3]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
dnn_regressor = train_dnn_regressor_model(
    optimizer,
    steps=2000,
    batch_size=100,
    hidden_units=[22,44,22,11],
    training_examples=training_features,
    training_targets=training_targets,
    validation_examples=validation_features,
    validation_targets=validation_targets)
numerical_fields = [
    'OverallQual', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea', 'MSSubClass'
]

categorical_fields = []


ret = convert_features(df, 
                       numerical_fields, 
                       categorical_fields, 
                       num_fields_proc = get_scaled_series(base = 1000.0), 
                       cat_fields_proc = get_dummies(dummy_na = True),
                       label_name = 'SalePrice',
                       train_validate_ratio = 0.7)


training_features = ret[0]
training_targets = ret[1]
validation_features = ret[2]
validation_targets = ret[3]
        
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
dnn_regressor = train_dnn_regressor_model(
    optimizer,
    steps=2000,
    batch_size=100,
    hidden_units=[22,44,22,11],
    training_examples=training_features,
    training_targets=training_targets,
    validation_examples=validation_features,
    validation_targets=validation_targets)
numerical_fields = [
    'OverallQual', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea', 'MSSubClass'
]

categorical_fields = []


ret = convert_features(df, 
                       numerical_fields, 
                       categorical_fields, 
                       num_fields_proc = get_logged_series, 
                       cat_fields_proc = get_dummies(dummy_na = True),
                       label_name = 'SalePrice',
                       train_validate_ratio = 0.7)


training_features = ret[0]
training_targets = ret[1]
validation_features = ret[2]
validation_targets = ret[3]
        
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
dnn_regressor = train_dnn_regressor_model(
    optimizer,
    steps=2000,
    batch_size=100,
    hidden_units=[22,44,22,11],
    training_examples=training_features,
    training_targets=training_targets,
    validation_examples=validation_features,
    validation_targets=validation_targets)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
dnn_regressor = train_dnn_regressor_model(
    optimizer,
    steps=400000,
    batch_size=2,
    hidden_units=[11,12,13,12,11,3],
    training_examples= training_features,
    training_targets=training_targets,
    validation_examples=validation_features,
    validation_targets=validation_targets)
optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
dnn_regressor = train_dnn_regressor_model(
    optimizer,
    steps=1000000,
    batch_size=10,
    hidden_units=[11,12,13,12,11,3],
    training_examples=training_features,
    training_targets=training_targets,
    validation_examples=validation_features,
    validation_targets=validation_targets)
numerical_fields = [
    'OverallQual', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea', 'MSSubClass'
]

categorical_fields = [
    'Neighborhood','MSZoning','KitchenQual','CentralAir','MasVnrType'
]


ret = convert_features(df, 
                       numerical_fields, 
                       categorical_fields, 
                       num_fields_proc = get_logged_series, 
                       cat_fields_proc = get_dummies(dummy_na = True),
                       label_name = 'SalePrice',
                       train_validate_ratio = 0.7)


training_features = ret[0]
training_targets = ret[1]
validation_features = ret[2]
validation_targets = ret[3]
optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
dnn_regressor = train_dnn_regressor_model(
    optimizer,
    steps=100000,
    batch_size=100,
    hidden_units=[11,12,13,12,11,3],
    training_examples=training_features,
    training_targets=training_targets,
    validation_examples=validation_features,
    validation_targets=validation_targets)
numerical_fields = [
    'OverallQual', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea', 'MSSubClass'
]

categorical_fields = [
    'Neighborhood','MSZoning','KitchenQual','CentralAir','MasVnrType'
]

ret = convert_features(df, 
                       numerical_fields, 
                       categorical_fields, 
                       num_fields_proc = get_logged_series, 
                       cat_fields_proc = get_dummies(dummy_na = True),
                       label_name = 'SalePrice',
                       train_validate_ratio = 1.0)

training_features_all = ret[0]
training_targets_all = ret[1]
optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
dnn_regressor = train_dnn_regressor_model(
    optimizer,
    steps=100000,
    batch_size=100,
    hidden_units=[11,12,13,12,11,3],
    training_examples=training_features_all,
    training_targets=training_targets_all,
    validation_examples=None,
    validation_targets=None)
numerical_fields = [
    'OverallQual', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'GrLivArea', 'FullBath', 'GarageCars', 'GarageArea', 'MSSubClass'
]

categorical_fields = [
    'Neighborhood','MSZoning','KitchenQual','CentralAir','MasVnrType'
]

ret = convert_features(df_test, 
                       numerical_fields, 
                       categorical_fields, 
                       num_fields_proc = get_logged_series, 
                       cat_fields_proc = get_dummies(dummy_na = True),
                       label_name = None,
                       train_validate_ratio = 1.0)

test_features = ret[0]

# haneld nan fields
for k in test_features.keys():
    s = test_features[k]
    na_cnt = sum(s.isna())
    if na_cnt > 0:
        test_features[k] = s.fillna(0.0)

predict_test_input_fn = lambda: my_input_fn_pred(test_features,
                                                 num_epochs=1, 
                                                 shuffle=False)

test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)
df_submit = pd.DataFrame()
df_submit['Id'] = df_test['Id']
df_submit['SalePrice'] = np.array([item['predictions'][0] for item in test_predictions])

df_submit.to_csv('./test_prediction.csv', index = False) 

