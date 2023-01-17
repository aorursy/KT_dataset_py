import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
COLUMNS = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
def normalize_data(data, is_test = False):

    # Extract columns

    columns = COLUMNS if is_test else COLUMNS + ['Survived']

    norm_data = data.loc[:, columns]

    

    # Quantify sex

    norm_data.loc[norm_data['Sex'] == 'male', 'Sex'] = 0

    norm_data.loc[norm_data['Sex'] == 'female', 'Sex'] = 1

    

    # Quantify embarked

    norm_data.loc[norm_data['Embarked'] == 'C', 'Embarked'] = 0

    norm_data.loc[norm_data['Embarked'] == 'Q', 'Embarked'] = 1

    norm_data.loc[norm_data['Embarked'] == 'S', 'Embarked'] = 2

    

    # Replace missing values

    column_medians = {column: norm_data[column].median() for column in COLUMNS}

    norm_data = norm_data.fillna(column_medians)

    

    return norm_data
normalize_data(train_data).head()
PURITY_CUTOFF = 0.6 # corresponds to a purity < 0.2 or > 0.8

SIZE_CUTOFF = 40

DEPTH_CUTOFF = 2



class Node(object):

    def GiniIndex(data1, data2):

        data1_survived = data1[data1.Survived == 1]

        data1_survived_ratio = float(len(data1_survived)) / len(data1)

        data1_died_ratio = 1.0 - data1_survived_ratio

        data1_gini = 1.0 - data1_survived_ratio * data1_survived_ratio - data1_died_ratio * data1_died_ratio



        data2_survived = data2[data2.Survived == 1]

        data2_survived_ratio = float(len(data2_survived)) / len(data2)

        data2_died_ratio = 1.0 - data2_survived_ratio

        data2_gini = 1.0 - data2_survived_ratio * data2_survived_ratio - data2_died_ratio * data2_died_ratio



        data1_ratio = float(len(data1)) / (len(data1) + len(data2))

        data2_ratio = 1.0 - data1_ratio



        return data1_gini * data1_ratio + data2_gini * data2_ratio

    

    def __init__(self, data, depth = 0):

        assert(len(data) > 0)

        print('- ' * depth + 'Depth: {0} Size: {1}'.format(depth, len(data)))

        self.data = data

        self.depth = depth

        self.purity = float(len(self.data[self.data['Survived'] == 1])) / len(self.data)

        self.is_branch = self.branch()

    

    def branch(self):

        if abs(2 * self.purity - 1) > PURITY_CUTOFF or len(self.data) < SIZE_CUTOFF or self.depth > DEPTH_CUTOFF:

            return False

        

        best_split = None

        for column in COLUMNS:

            left = self.data.sort_values(by = column).reset_index(drop = True)

            right = pd.DataFrame(columns = left.columns)

            for i in range(len(left) - 1, 0, -1):

                left_val = left.iloc[i - 1][column]

                right_val = left.iloc[i][column]

                right = right.append(left.iloc[i], ignore_index = True)

                left = left.drop(i)

                if left_val == right_val:

                    continue

                split = (left_val + right_val) / 2

                gini_index = Node.GiniIndex(left, right)

                if best_split is None or gini_index < best_split['gini_index']:

                    best_split = {'column': column, 'split': split, 'gini_index': gini_index, 'left': left, 'right': right}

        

        self.column = best_split['column']

        self.split = best_split['split']

        self.gini_index = best_split['gini_index']

        

        #print(len(data1), len(data2))

        self.left = Node(best_split['left'], self.depth + 1)

        self.right = Node(best_split['right'], self.depth + 1)

        

        return True

        

    def feed(self, passenger):

        if self.is_branch:

            if passenger[self.column] < self.split:

                return self.left.feed(passenger)

            else:

                return self.right.feed(passenger)

        else:

            return self.purity

        

    def __str__(self):

        s = ''

        prefix = '\t' * self.depth

        s += prefix + ('[BRANCH]' if self.is_branch else '[LEAF]') + '\n'

        s += prefix + ' > Size: ' + str(len(self.data)) + '\n'

        s += prefix + ' > Purity: ' + str(self.purity) + '\n'

        if self.is_branch:

            s += prefix + ' > Column: ' + self.column + '\n'

            s += prefix + ' > Split: ' + str(self.split) + '\n'

            s += prefix + ' > Gini Index: ' + str(self.gini_index) + '\n'

            s += str(self.left)

            s += str(self.right)

        return s

    

    def __repr__(self):

        return str(self)
norm_train_data = normalize_data(train_data)



tree_data = norm_train_data.sample(frac = 0.75, random_state = 1).reset_index(drop = True)

eval_data = norm_train_data.drop(tree_data.index).reset_index(drop = True)
tree_data.head()
eval_data.head()
tree = Node(tree_data)

tree
def make_predictions(data, tree):

    predictions = []

    for _, row in data.iterrows():

        purity = tree.feed(row)

        if purity < 0.5:

            predictions.append(0)

        else:

            predictions.append(1)

    return predictions
train_predictions = make_predictions(eval_data, tree)

n_correct = 0

for i in range(len(train_predictions)):

    if train_predictions[i] == eval_data['Survived'][i]:

        n_correct += 1

print(float(n_correct) / len(train_predictions))
final_tree = Node(norm_train_data)

final_tree
norm_test_data = normalize_data(test_data, True)

test_predictions = make_predictions(norm_test_data, final_tree)

test_data['Survived'] = test_predictions

test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)