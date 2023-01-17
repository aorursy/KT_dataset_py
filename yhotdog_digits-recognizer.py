import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
%matplotlib inline
IMAGE_SIZE = 28

def split(data, ratio, seed = 42):
    np.random.seed(seed)
    is_train = np.random.random(len(data)) <= ratio
    train, test = data[is_train == True], data[is_train == False]
    return train, test

def show_image(data, row_index, label_name = 'label'):
    row = data[row_index : row_index + 1]
    print(row[label_name].values[0])
    plt.imshow(row[row.columns[1:]].values.reshape(IMAGE_SIZE, IMAGE_SIZE))
    
def get_pixel_feature_names(data):
    return list(filter(lambda name : name.startswith('pixel'), data.columns))
    
def predictor_random_forest(train, test, feature_names, forest_args = {}):
    clf = RandomForestClassifier(**forest_args)
    clf.fit(train[feature_names], train['label'])
    return clf.predict(test[feature_names])
    
def predictor_random_forest_pixels(train, test):
    return predictor_random_forest(train, test, get_pixel_feature_names(train))
    
def predictor_random_forest_pixels_many_trees(train, test):
    return predictor_random_forest(train, test, get_pixel_feature_names(train), {"n_estimators": 25})



def calc_image_mean(image, axis):
    return pd.Series(image.reshape(IMAGE_SIZE, IMAGE_SIZE).mean(axis))

def calc_data_means(data, axis):
    return data[get_pixel_feature_names(data)].apply(lambda image : calc_image_mean(image, axis), axis = 'columns')

def calc_data_with_rows_means_and_cols_mean(data):
    cols_means = calc_data_means(data, axis = 0)
    cols_means = cols_means.rename(columns = lambda col_index : 'col_mean_' + str(col_index))
    rows_means = calc_data_means(data, axis = 1)
    rows_means = rows_means.rename(columns = lambda row_index : 'row_mean_' + str(row_index))
    return pd.concat([data, rows_means, cols_means], axis = 1)
    
def get_mean_feature_names(data):
    return list(filter(lambda name : name.startswith('col_mean_') or name.startswith('row_mean_'), data.columns))

def predictor_random_forest_pixels_and_means(train, test):
    return predictor_random_forest(train, test, get_pixel_feature_names(train) + get_mean_feature_names(train))

def predictor_random_forest_means(train, test):
    return predictor_random_forest(train, test, get_mean_feature_names(train))





def benchmark(data, predictor):
    scores = []
    times = []
    for i in range(5):
        train, test = split(data, 0.8)
        start_time = time.time()
        test_predictions = predictor(train, test)
        times.append(time.time() - start_time)
        correct_classifications = (test['label'] == test_predictions)
        scores.append(round(100.0 * len(correct_classifications[correct_classifications]) / len(test), 2))
        #pd.crosstab(test['label'], test_predictions, rownames=['actual'], colnames=['preds'])
    print('average score ' + str(sum(scores) / len(scores)) + \
        ' calculated in ' + str(round(sum(times) / len(times), 2)) + ' seconds using ' + predictor.__name__)
        
def solve(csv_path, data, unknown, predictor):
    unknown['Label'] = predictor(data, unknown)
    unknown.index += 1
    unknown['Label'].to_csv(csv_path, header = True, index_label = 'ImageId')
data = calc_data_with_rows_means_and_cols_mean(pd.read_csv('../input/train.csv'))
unknown = calc_data_with_rows_means_and_cols_mean(pd.read_csv('../input/test.csv'))
benchmark(data, predictor_random_forest_pixels)
benchmark(data, predictor_random_forest_pixels_and_means)
benchmark(data, predictor_random_forest_means)
benchmark(data, predictor_random_forest_pixels_many_trees)
#solve('submission.csv', data, unknown, predictor_random_forest_simple)
