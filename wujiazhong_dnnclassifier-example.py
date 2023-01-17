#!/usr/bin/python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split

data = pd.DataFrame(pd.read_csv("../input/car.data.csv"))
buying_mapping = {
    'vhigh': 0,
    'high': 1,
    'med': 2,
    'low': 3
}

maint_mapping = {
    'vhigh': 0,
    'high': 1,
    'med': 2,
    'low': 3
}

doors_mapping = {
    '2': 0,
    '3': 1,
    '4': 2,
    '5more': 3
}

persons_mapping = {
    '2': 0,
    '4': 1,
    'more': 2
}

lug_boot_mapping = {
    'small': 0,
    'med': 1,
    'big': 2
}

safety_mapping = {
    'low': 0,
    'med': 1,
    'high': 2
}

rating_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

data['buying'] = data['buying'].map(buying_mapping)
data['maint'] = data['maint'].map(maint_mapping)
data['doors'] = data['doors'].map(doors_mapping)
data['persons'] = data['persons'].map(persons_mapping)
data['lug_boot'] = data['lug_boot'].map(lug_boot_mapping)
data['safety'] = data['safety'].map(safety_mapping)
data['rating'] = data['rating'].map(rating_mapping)

X_train, X_test, y_train, y_test = train_test_split(data[["buying", "maint", "doors", "persons", "lug_boot", "safety"]].values,
                                                    data["rating"].values, random_state=42)


def main():
    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=6)]

    # Build 3 layer DNN with 512, 256, 128 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[512, 256, 128],
                                                n_classes=4,
                                                optimizer=tf.train.ProximalAdagradOptimizer(
                                                    learning_rate=0.15,
                                                    l1_regularization_strength=0.001
                                                ))

    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(X_train)
        y = tf.constant(y_train)
        return x, y

    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=1200)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(X_test)
        y = tf.constant(y_test)

        return x, y

    # Evaluate accuracy.
    # print(classifier.evaluate(input_fn=get_test_inputs, steps=1))
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
    graph_location = '/tmp/tensorflow/car-evaluation'
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    print("Test Accuracy: {0:f}".format(accuracy_score))

    # Classify two new flower samples.
    # med,med,5more,more,med,high,vgood
    # med,med,4,2,small,high,unacc
    def new_samples():
        return np.array([[2, 2, 3, 2, 1, 2], [2, 2, 2, 0, 0, 2]], dtype=np.float32)

    predictions = classifier.predict(input_fn=new_samples)

    print("New Samples, Class Predictions: {}".format(predictions))


if __name__ == "__main__":
    main()

