# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from io import StringIO;

import os;

import os.path;



def parse_header_of_csv(headline):

    # Isolate the headline columns:

#    headline = csv_str[:csv_str.index('\n')];

    columns = headline.split(',');



    # The first column should be timestamp:

    assert columns[0] == 'timestamp';

    # The last column should be label_source:

    assert columns[-1] == 'label_source';

    

    # Search for the column of the first label:

    for (ci,col) in enumerate(columns):

        if col.startswith('label:'):

            first_label_ind = ci;

            break;

        pass;



    # Feature columns come after timestamp and before the labels:

    feature_names = columns[1:first_label_ind];

    # Then come the labels, till the one-before-last column:

    label_names = columns[first_label_ind:-1];

    for (li,label) in enumerate(label_names):

        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:

        assert label.startswith('label:');

        label_names[li] = label.replace('label:','');

        pass;

    

    return (feature_names,label_names);



def parse_body_of_csv(csv_str,n_features):

    # Read the entire CSV body into a single numeric matrix:

    full_table = np.loadtxt(StringIO(csv_str),delimiter=',');#,skiprows=1);

    

    # Timestamp is the primary key for the records (examples):

    timestamps = full_table[:,0].astype(int);

    

    # Read the sensor features:

    X = full_table[:,1:(n_features+1)];

    

    # Read the binary label values, and the 'missing label' indicators:

    trinary_labels_mat = full_table[:,(n_features+1):-1]; # This should have values of either 0., 1. or NaN

    M = np.isnan(trinary_labels_mat); # M is the missing label matrix

    Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix

    

    return (X,Y,M,timestamps);



'''

Interpret the feature names to figure out for each feature what is the sensor it was extracted from.

'''

def get_sensor_names_from_features(feature_names):

    feat_sensor_names = np.array([None for feat in feature_names]);

    for (fi,feat) in enumerate(feature_names):

        if feat.startswith('raw_acc'):

            feat_sensor_names[fi] = 'Acc';

            pass;

        elif feat.startswith('proc_gyro'):

            feat_sensor_names[fi] = 'Gyro';

            pass;

        elif feat.startswith('raw_magnet'):

            feat_sensor_names[fi] = 'Magnet';

            pass;

        elif feat.startswith('watch_acceleration'):

            feat_sensor_names[fi] = 'WAcc';

            pass;

        elif feat.startswith('watch_heading'):

            feat_sensor_names[fi] = 'Compass';

            pass;

        elif feat.startswith('location'):

            feat_sensor_names[fi] = 'Loc';

            pass;

        elif feat.startswith('location_quick_features'):

            feat_sensor_names[fi] = 'Loc';

            pass;

        elif feat.startswith('audio_naive'):

            feat_sensor_names[fi] = 'Aud';

            pass;

        elif feat.startswith('audio_properties'):

            feat_sensor_names[fi] = 'AP';

            pass;

        elif feat.startswith('discrete'):

            feat_sensor_names[fi] = 'PS';

            pass;

        elif feat.startswith('lf_measurements'):

            feat_sensor_names[fi] = 'LF';

            pass;

        else:

            raise ValueError("!!! Unsupported feature name: %s" % feat);



        pass;



    return feat_sensor_names;  



'''

Read the data (precomputed sensor-features and labels) for a user.

This function assumes the user's data file is present.

'''

def read_user_data(uuid):

    user_data_file = os.path.join('../input','%s.features_labels.csv' % uuid);



    # Read the entire csv file of the user:

    with open(user_data_file,'r') as fid:

        csv_headline = fid.readline().strip();

        csv_body = fid.read();

        pass;



    (feature_names,label_names) = parse_header_of_csv(csv_headline);

    feat_sensor_names = get_sensor_names_from_features(feature_names);

    n_features = len(feature_names);

    (X,Y,M,timestamps) = parse_body_of_csv(csv_body,n_features);



    return (X,Y,M,timestamps,feature_names,feat_sensor_names,label_names);



def validate_column_names_are_consistent(old_column_names,new_column_names):

    if len(old_column_names) != len(new_column_names):

        raise ValueError("!!! Inconsistent number of columns.");

        

    for ci in range(len(old_column_names)):

        if old_column_names[ci] != new_column_names[ci]:

            raise ValueError("!!! Inconsistent column %d) %s != %s" % (ci,old_column_names[ci],new_column_names[ci]));

        pass;

    return;



def read_multiple_users_data(uuids):

    feature_names = None;

    feat_sensor_names = None;

    label_names = None;

    X_parts = [];

    Y_parts = [];

    M_parts = [];

    timestamps_parts = [];

    uuid_inds_parts = [];

    for (ui,uuid) in enumerate(uuids):

        (X_i,Y_i,M_i,timestamps_i,feature_names_i,feat_sensor_names_i,label_names_i);

        # Make sure the feature names are consistent among all users:

        if feature_names is None:

            feature_names = feature_names_i;

            feat_sensor_names = feat_sensor_names_i;

            pass;

        else:

            validate_column_names_are_consistent(feature_names,feature_names_i);

            pass;

        # Make sure the label names are consistent among all users:

        if label_names is None:

            label_names = label_names_i;

            pass;

        else:

            validate_column_names_are_consistent(label_names,label_names_i);

            pass;

        # Accumulate this user's data:

        X_parts.append(X_i);

        Y_parts.append(Y_i);

        M_parts.append(M_i);

        timestamps_parts.append(timestamps_i);

        uuid_inds_parts.append(ui*np.ones(len(timestamps_i)));

        pass;

    

    # Combine all the users' data:

    X = np.concatenate(tuple(X_parts),axis=0);

    Y = np.concatenate(tuple(Y_parts),axis=0);

    M = np.concatenate(tuple(M_parts),axis=0);

    timestamps = np.concatenate(tuple(timestamps_parts),axis=0);

    uuid_inds = np.concatenate(tuple(uuid_inds_parts),axis=0);

    

    return (X,Y,M,uuid_inds,timestamps,feature_names,feat_sensor_names,label_names);

# Let's load data for the two users:

(X1,Y1,M1,timestamps1,feature_names,feat_sensor_names,label_names) = read_user_data('user1');

(X2,Y2,M2,timestamps2,feature_names2,feat_sensor_names2,label_names2) = read_user_data('user2');



# The columns of all the users' files should be consistent,

# but if you want to make sure, you can validate it:

validate_column_names_are_consistent(feature_names,feature_names2);

validate_column_names_are_consistent(label_names,label_names2);
print("User 1 has %d examples. User 2 has %d examples" % (X1.shape[0],X2.shape[0]));

print("There are %d sensor-features from %d sensors in the data" % (len(feature_names),len(set(feat_sensor_names))));

print("There is information about %d context-labels" % len(label_names));
import sklearn.linear_model;



def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):

    use_feature = np.zeros(len(feat_sensor_names),dtype=bool);

    for sensor in sensors_to_use:

        is_from_sensor = (feat_sensor_names == sensor);

        use_feature = np.logical_or(use_feature,is_from_sensor);

        pass;

    X = X[:,use_feature];

    return X;



def estimate_standardization_params(X_train):

    mean_vec = np.nanmean(X_train,axis=0);

    std_vec = np.nanstd(X_train,axis=0);

    return (mean_vec,std_vec);



def standardize_features(X,mean_vec,std_vec):

    # Subtract the mean, to centralize all features around zero:

    X_centralized = X - mean_vec.reshape((1,-1));

    # Divide by the standard deviation, to get unit-variance for all features:

    # * Avoid dividing by zero, in case some feature had estimate of zero variance

    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1));

    X_standard = X_centralized / normalizers;

    return X_standard;



def train_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label):

    # Project the feature matrix to the features from the desired sensors:

    X_train = project_features_to_selected_sensors(X_train,feat_sensor_names,sensors_to_use);

    print("== Projected the features to %d features from the sensors: %s" % (X_train.shape[1],', '.join(sensors_to_use)));



    # It is recommended to standardize the features (subtract mean and divide by standard deviation),

    # so that all their values will be roughly in the same range:

    (mean_vec,std_vec) = estimate_standardization_params(X_train);

    X_train = standardize_features(X_train,mean_vec,std_vec);

    

    # The single target label:

    label_ind = label_names.index(target_label);

    y = Y_train[:,label_ind];

    missing_label = M_train[:,label_ind];

    existing_label = np.logical_not(missing_label);

    

    # Select only the examples that are not missing the target label:

    X_train = X_train[existing_label,:];

    y = y[existing_label];

    

    # Also, there may be missing sensor-features (represented in the data as NaN).

    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).

    # You can also further select examples - only those that have values for all the features.

    # For this tutorial, let's use the simple heuristic of zero-imputation:

    X_train[np.isnan(X_train)] = 0.;

    

    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \

          (len(y),target_label,sum(y),sum(np.logical_not(y))) );

    

    # Now, we have the input features and the ground truth for the output label.

    # We can train a logistic regression model.

    

    # Typically, the data is highly imbalanced, with many more negative examples;

    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:

    lr_model = sklearn.linear_model.LogisticRegression(class_weight='balanced');

    lr_model.fit(X_train,y);

    

    # Assemble all the parts of the model:

    model = {\

            'sensors_to_use':sensors_to_use,\

            'target_label':target_label,\

            'mean_vec':mean_vec,\

            'std_vec':std_vec,\

            'lr_model':lr_model};

    

    return model;
# Let's train a model to use the features from 

# the phone accelerometer (Acc) and the watch-accelerometer (WAcc)

# to predict the label 'Walking':

sensors_to_use = ['Acc','WAcc'];

target_label = 'FIX_walking'; # this is just code name for the cleaned version of the label 'Walking'

model = train_model(X1,Y1,M1,feat_sensor_names,label_names,sensors_to_use,target_label);
%matplotlib inline

import matplotlib as mpl;

import matplotlib.pyplot as plt;



def test_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model):

    # Project the feature matrix to the features from the sensors that the classifier is based on:

    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use']);

    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],', '.join(model['sensors_to_use'])));



    # We should standardize the features the same way the train data was standardized:

    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec']);

    

    # The single target label:

    label_ind = label_names.index(model['target_label']);

    y = Y_test[:,label_ind];

    missing_label = M_test[:,label_ind];

    existing_label = np.logical_not(missing_label);

    

    # Select only the examples that are not missing the target label:

    X_test = X_test[existing_label,:];

    y = y[existing_label];

    timestamps = timestamps[existing_label];



    # Do the same treatment for missing features as done to the training data:

    X_test[np.isnan(X_test)] = 0.;

    

    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." % \

          (len(y),model['target_label'],sum(y),sum(np.logical_not(y))) );

    

    # Preform the prediction:

    y_pred = model['lr_model'].predict(X_test);

    

    # Naive accuracy (correct classification rate):

    accuracy = np.mean(y_pred == y);

    

    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:

    tp = np.sum(np.logical_and(y_pred,y));

    tn = np.sum(np.logical_and(np.logical_not(y_pred),np.logical_not(y)));

    fp = np.sum(np.logical_and(y_pred,np.logical_not(y)));

    fn = np.sum(np.logical_and(np.logical_not(y_pred),y));

    

    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):

    sensitivity = float(tp) / (tp+fn);

    specificity = float(tn) / (tn+fp);

    

    # Balanced accuracy is a more fair replacement for the naive accuracy:

    balanced_accuracy = (sensitivity + specificity) / 2.;

    

    # Precision:

    # Beware from this metric, since it may be too sensitive to rare labels.

    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,

    # and for each label the pos/neg ratio is different.

    # This can cause undesirable and misleading results when averaging precision across different labels.

    precision = float(tp) / (tp+fp);

    

    print("-"*10);

    print('Accuracy*:         %.2f' % accuracy);

    print('Sensitivity (TPR): %.2f' % sensitivity);

    print('Specificity (TNR): %.2f' % specificity);

    print('Balanced accuracy: %.2f' % balanced_accuracy);

    print('Precision**:       %.2f' % precision);

    print("-"*10);

    

    print('* The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives).')

    print('** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels.')

    

    fig = plt.figure(figsize=(10,4),facecolor='white');

    ax = plt.subplot(1,1,1);

    ax.plot(timestamps[y],1.4*np.ones(sum(y)),'|g',markersize=10,label='ground truth');

    ax.plot(timestamps[y_pred],np.ones(sum(y_pred)),'|b',markersize=10,label='prediction');

    

    seconds_in_day = (60*60*24);

    tick_seconds = range(timestamps[0],timestamps[-1],seconds_in_day);

    tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int);

    

    ax.set_ylim([0.5,5]);

    ax.set_xticks(tick_seconds);

    ax.set_xticklabels(tick_labels);

    plt.xlabel('days of participation',fontsize=14);

    ax.legend(loc='best');

    plt.title('%s\nGround truth vs. predicted' % model['target_label']);

    

    return;
# First, let's test the goodness-of-fit - how well the model fit to the training data (user1's data):

test_model(X1,Y1,M1,timestamps1,feat_sensor_names,label_names,model);
# Now, let's look at the more important evaluation - generalization.

# Meaning, how well the model predicts the context for unseen people (test on user2):

test_model(X2,Y2,M2,timestamps2,feat_sensor_names,label_names,model);