# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from collections import Counter



import numpy as np

from numpy import genfromtxt

import scipy.io

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



import random





# Any results you write to the current directory are saved as output.
np.random.seed(0)
path_train = '../input/data/datasets/titanic/titanic_training.csv'

titanic_train = pd.read_csv(path_train)
path_test = '../input/data/datasets/titanic/titanic_testing_data.csv'

titanic_test = pd.read_csv(path_test)
titanic_train
titanic_train = titanic_train.drop(705)

y = titanic_train['survived'].values  # label = survived
y
titanic_test
dropped_train = titanic_train.drop(['ticket', 'cabin'], axis=1)

dropped_test = titanic_test.drop(['ticket', 'cabin'], axis=1)
dropped_train
dropped_train[dropped_train['survived'].isnull()]
dropped_train['pclass'].fillna((dropped_train['pclass'].mode()), inplace=True)

dropped_train['sex'].fillna((dropped_train['sex'].mode()), inplace=True)

dropped_train['age'].fillna((dropped_train['age'].median()), inplace=True)

dropped_train['sibsp'].fillna((dropped_train['sibsp'].mode()), inplace=True)

dropped_train['parch'].fillna((dropped_train['parch'].mode()), inplace=True)

dropped_train['fare'].fillna((dropped_train['fare'].median()), inplace=True)

dropped_train['embarked'].fillna((dropped_train['embarked'].mode()[0]), inplace=True)
dropped_train[dropped_train.isna().values]
dropped_train.shape
dropped_test['pclass'].fillna((dropped_test['pclass'].mode()), inplace=True)

dropped_test['sex'].fillna((dropped_test['sex'].mode()), inplace=True)

dropped_test['age'].fillna((dropped_test['age'].median()), inplace=True)

dropped_test['sibsp'].fillna((dropped_test['sibsp'].mode()), inplace=True)

dropped_test['parch'].fillna((dropped_test['parch'].mode()), inplace=True)

dropped_test['fare'].fillna((dropped_test['fare'].median()), inplace=True)

dropped_test['embarked'].fillna((dropped_test['embarked'].mode()[0]), inplace=True)
dropped_test[dropped_test.isna().values]
combined = pd.concat((dropped_train[dropped_test.columns], dropped_test))
onehot = pd.get_dummies(combined, columns=['sex', 'pclass', 'embarked'])
onehot_train = onehot[:dropped_train.shape[0]]
onehot_test = onehot[dropped_train.shape[0]:]
onehot_test
"""

This is the starter code and some suggested architecture we provide you with.

But feel free to do any modifications as you wish or just completely ignore

all of them and have your own implementations.

"""

from collections import Counter



import numpy as np

from numpy import genfromtxt

import scipy.io

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



import random





class Node:

    def __init__(self, split_rule, left, right, label=None):

        self.split_rule = split_rule

        self.left = left

        self.right = right

        self.label = label



    def __repr__(self):

        return "split_rule: {}\nleft: {}\nright: {}\nlabel: {}".format(self.split_rule, self.left,

                                             repr(self.right), self.label)



class DecisionTree:

    def __init__(self, max_depth=float('inf')):

        """

        TODO: initialization of a decision tree

        """

        self.max_depth = max_depth



    @staticmethod

    def distribution(y):

        """Computes distribution of labels"""

        counter = Counter(y)

        return np.array([counter[key] / len(y) for key in counter])



    @staticmethod

    def split_gain(parent_purity, left_child_labels, right_child_labels, metric):

        weight_left = len(left_child_labels) / (len(left_child_labels) +

                                                len(right_child_labels))

        weight_right = 1 - weight_left



        return parent_purity - (weight_left * metric(left_child_labels) +

                                weight_right * metric(right_child_labels))



    @staticmethod

    def fast_split_gain(left_0_count, left_1_count,

                        right_0_count, right_1_count):

        left_total = left_0_count + left_1_count

        right_total = right_0_count + right_1_count



        weight_left = left_total / (left_total + right_total)

        weight_right = 1 - weight_left



        left_0_prob = left_0_count / left_total

        left_1_prob = 1 - left_0_prob



        right_0_prob = right_0_count / right_total

        right_1_prob = 1 - right_0_prob



        left_impurity = 1 - left_0_prob ** 2 - left_1_prob ** 2

        right_impurity = 1 - right_0_prob ** 2 - right_1_prob ** 2



        return (weight_left * left_impurity + weight_right * right_impurity)



    @staticmethod

    def purification_gain(X, y, thresh, metric):

        left_mask = X < thresh

        left_child_labels = y[left_mask]

        right_child_labels = y[~left_mask]



        weight_left = len(left_child_labels) / len(y)

        weight_right = 1 - weight_left



        parent_entropy = metric(y)



        return parent_entropy - (weight_left * metric(left_child_labels) +

                                 weight_right * metric(right_child_labels))



    @staticmethod

    def entropy(y):

        """

        TODO: implement a method that calculates the entropy given all the labels

        """

        probs = DecisionTree.distribution(y)



        return scipy.stats.entropy(probs)



    @staticmethod

    def information_gain(X, y, thresh):

        """

        TODO: implement a method that calculates information gain given a vector of features

        and a split threshold

        """

        return DecisionTree.purification_gain(X, y, thresh, DecisionTree.entropy)



    @staticmethod

    def gini_impurity_from_dist(probs):

        return 1 - sum(probs ** 2)



    @staticmethod

    def gini_impurity(y):

        """

        TODO: implement a method that calculates the gini impurity given all the labels

        """

        probs = DecisionTree.distribution(y)

        return DecisionTree.gini_impurity_from_dist(probs)



    @staticmethod

    def gini_purification(X, y, thresh):

        """

        TODO: implement a method that calculates reduction in impurity gain given a vector of features

        and a split threshold

        """

        return DecisionTree.purification_gain(X, y, thresh, DecisionTree.gini_impurity)



    def split(self, X, y, idx, thresh):

        """

        TODO: implement a method that return a split of the dataset given an index of the feature and

        a threshold for it

        """

        left_mask = X[:, idx] < thresh

        left_y = y[left_mask]

        left_X = X[left_mask, :]



        right_mask = ~left_mask

        right_y = y[right_mask]

        right_X = X[right_mask, :]



        return (left_X, left_y), (right_X, right_y)



    def segmenter(self, X, y, feature_subset=None):

        """

        TODO: compute entropy gain for all single-dimension splits,

        return the feature and the threshold for the split that

        has maximum gain

        """

        min_purity = float('inf')

        best_feature = None

        best_threshold = None



        features = range(X.shape[1]) if feature_subset is None else feature_subset



        total_0 = sum(y == 0)

        total_1 = len(y) - total_0



        for feature_index in features:

            feature_col = X[:, feature_index]

            sorted_indices = np.argsort(feature_col)



            left_0_count = feature_col[sorted_indices[0]] == 0

            left_1_count = 1 - left_0_count

            right_0_count = total_0 - left_0_count

            right_1_count = total_1 - left_1_count



            for i in range(len(sorted_indices) - 1):

                if feature_col[sorted_indices[i]] == feature_col[sorted_indices[i + 1]]:

                    new_0 = y[sorted_indices[i + 1]] == 0

                    new_1 = 1 - new_0



                    left_0_count += new_0

                    left_1_count += new_1



                    right_0_count -= new_0

                    right_1_count -= new_1

                    continue



                threshold = (feature_col[sorted_indices[i]] +

                             feature_col[sorted_indices[i + 1]]) / 2



                split_purity = DecisionTree.fast_split_gain(left_0_count, left_1_count, right_0_count, right_1_count)



                if split_purity < min_purity:

                    min_purity = split_purity

                    best_feature = feature_index

                    best_threshold = threshold



        return best_feature, best_threshold



    def fit(self, X, y, depth=0, random=False):

        """

        TODO: fit the model to a training set. Think about what would be

        your stopping criteria

        """

        if len(y) == 0:

            return Node(None, None, None, None)



        if depth == self.max_depth or len(set(y)) == 1:

            return Node(None, None, None, scipy.stats.mode(y)[0][0])



        feature_subset = None



        if random:

            feature_subset = np.random.choice(range(X.shape[1]),

                                              int(round(np.sqrt(X.shape[1]))),

                                              replace=False)



        best_feature, best_threshold = self.segmenter(X, y, feature_subset)



        if best_threshold is None:

            return Node(None, None, None, scipy.stats.mode(y)[0][0])



        left_split, right_split = self.split(X, y, best_feature, best_threshold)



        left_node = self.fit(left_split[0], left_split[1], depth + 1, random)

        right_node = self.fit(right_split[0], right_split[1], depth + 1, random)



        parent_node = Node((best_feature, best_threshold), left_node, right_node)

        self.root = parent_node

        return parent_node



    def predict(self, X):

        """

        TODO: predict the labels for input data

        """

        labels = []



        for point in X:

            current = self.root



            while current.label is None:

                feature, threshold = current.split_rule

                current = current.left if point[feature] < threshold else current.right



            labels.append(current.label)



        return np.array(labels)



    def __repr__(self):

        """

        TODO: one way to visualize the decision tree is to write out a __repr__ method

        that returns the string representation of a tree. Think about how to visualize

        a tree structure. You might have seen this before in CS61A.

        """

        return repr(self.root)
class RandomForest():



    def __init__(self, n=10):

        """

        TODO: initialization of a random forest

        """

        self.forest = []

        self.n = n



        for _ in range(self.n):

            self.forest.append(DecisionTree())



    def fit(self, X, y):

        """

        TODO: fit the model to a training set.

        """

        for tree in self.forest:

            sample = np.random.choice(range(len(y)), len(y), replace=True)



            tree.fit(X[sample, :], y[sample], random=True)



    def predict(self, X):

        """

        TODO: predict the labels for input data

        """

        predictions = []



        for tree in self.forest:

            tree_predictions = tree.predict(X)



            predictions.append(tree_predictions.reshape(X.shape[0], 1))



        concatenated = np.concatenate(predictions, axis=1)



        return scipy.stats.mode(concatenated, axis=1)[0].flatten()
onehot_train_values = onehot_train.values
X_train, X_test, y_train, y_test = train_test_split(onehot_train_values, y)
tree = DecisionTree(onehot_train.shape[1])

tree.fit(X_train, y_train)

tree_train_predictions = tree.predict(X_train)

tree_test_predictions = tree.predict(X_test)



forest = RandomForest(onehot_train.shape[1])

forest.fit(X_train, y_train)

forest_train_predictions = forest.predict(X_train)

forest_test_predictions = forest.predict(X_test)



print("tree train accuracy:", sum(tree_train_predictions == y_train) / len(tree_train_predictions))

print("tree test accuracy:", sum(tree_test_predictions == y_test) /

      len(tree_test_predictions))



print("forest train accuracy:", sum(forest_train_predictions == y_train) / len(forest_train_predictions))

print("forest test accuracy:", sum(forest_test_predictions == y_test) / len(forest_test_predictions))
# Usage results_to_csv(clf.predict(X_test))

def results_to_csv(y_test, name):

    y_test = y_test.astype(int)

    df = pd.DataFrame({'Category': y_test})

    df.index += 1  # Ensures that the index starts at 1. 

    df.to_csv(f'{name}.csv', index_label='Id')
tree = DecisionTree(onehot_train.shape[1])

tree.fit(onehot_train_values, y)

tree_test_predictions = tree.predict(onehot_test.values)



forest = RandomForest(onehot_train.shape[1])

forest.fit(onehot_train_values, y)

forest_test_predictions = forest.predict(onehot_test.values)



results_to_csv(tree_test_predictions, 'titanic_tree')

results_to_csv(forest_test_predictions, 'titanic_forest')