#import required libraries
import numpy as np 
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
#function to create a quality label
def QualityLabeller(data):
    data.loc[:,'quality_label'] = np.where(data.loc[:,'quality']>5, 1, 0)
    return data
#function to scale data
def DataScaler(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data
#main function
def kernel():
    raw_data = pd.read_csv('../input/winequality-red.csv')
    raw_data = pd.DataFrame(raw_data)
    all_features = list(raw_data)
    target = ['quality']
    features = list(set(all_features)-set(target))
    raw_data.loc[:,features] = DataScaler(raw_data.loc[:,features])

    labelled_data = QualityLabeller(raw_data)
    target.append('quality_label')

    train_data = labelled_data.sample(250)
    test_data = labelled_data.drop(train_data.index)
    x_train = train_data.drop(target,axis=1)
    y_train = train_data.loc[:,'quality_label']

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    
    wine_quality_classifier = MLPClassifier(
                                            solver='lbfgs',
                                            alpha=1e-5, 
                                            hidden_layer_sizes=(13,10,5),
                                            random_state=1,
                                            max_iter=1000
                                            )

    wine_quality_classifier.fit(x_train, y_train)

    predicted_quality_label = wine_quality_classifier.predict(np.asarray(test_data.drop(target,axis=1)))
    test_data.loc[:,'predicted_quality_label'] = predicted_quality_label
    accuracy = float(len(test_data.loc[test_data['quality_label'] == test_data['predicted_quality_label'],:]))/float(len(test_data))
    return accuracy
kernel()