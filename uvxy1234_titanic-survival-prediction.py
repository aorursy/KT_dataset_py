import numpy as np

import pandas as pd

import math

from sklearn.preprocessing import LabelEncoder

from sklearn.semi_supervised import label_propagation

from scipy import stats



from keras.models import Sequential

from keras.optimizers import SGD, RMSprop, Adam

from keras.layers import Dense, Activation, Dropout

from keras.callbacks import EarlyStopping

from keras import regularizers



from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV



from matplotlib import pyplot as plt

import seaborn as sns





%matplotlib inline 

%config InlineBackend.figure_format = 'retina' ## retina display. 

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input/")) 
## Importing the datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
## Check Dimensions

Size_train = train.shape[0]

Size_test = test.shape[0]

Dimension = train.shape[1] - 1

print("Training set size:{}, Testing set size:{}, Feature Dimension:{}".format(Size_train, Size_test, Dimension))
#print out the first 5 rows of training set

train.head()
# define plotting tool to check distribution

def plot_distribution_by_label(X, val_str, label_str, label_dict, missing=-20):

    fig, axes = plt.subplots(2,1,figsize=(8,6))

    sns.set_style('white')

    sns.distplot(X[val_str].fillna(missing), rug=True, color='b', ax=axes[0])

    ax0 = axes[0]

    ax0.set_title(val_str +' distribution')

    ax0.set_xlabel('')

    

    



    ax1 = axes[1]

    ax1.set_title(val_str +' survived distribution')

    k1 = sns.distplot(X[X[label_str]==0][val_str].fillna(missing), hist=False, color='r', ax=ax1, label=label_dict[0])

    k2 = sns.distplot(X[X[label_str]==1][val_str].fillna(missing), hist=False, color='g', ax=ax1, label=label_dict[1])

    ax1.set_xlabel('')



    ax1.legend(fontsize=16)
label_str = 'Survived'

label_dict = {0:'dead', 1:'alive'}



val_strs = ['Age','Pclass','SibSp','Parch','Fare']

for val_str in val_strs:    

    plot_distribution_by_label(train, val_str, label_str, label_dict)
# split the labels out.



X_train = train.drop(['Survived'], axis=1)

Y_train = train[['PassengerId','Survived']]

X_test = test

#X_test = test.drop(['Name'], axis=1)

X_train.head()
X_train.info()
X_test.info()
X_train.describe()
columns_train = X_train.drop(['PassengerId'], axis=1).columns

for column_train in columns_train:

    print("{0}: {1}".format(column_train, X_train[column_train].unique()))
# pd.series(xxx).isna() return the boolings if it's nan/none or not

X_train[X_train["Cabin"].isna()].head()
X_train.isna().sum()
# Merge the train/test temporarily to do the feature engineering. 

# Also the reason for merge is to ensure the distribution between train/test sets are the same.

X_temp = pd.concat([X_train, X_test], sort = False)



# shows the count of missing value for each column

missing_init = -20

X_temp.isna().sum()

# Extract title and last name from Name

# Extract length of len, which is likely correlates to 'survived'

X_temp_title = X_temp['Name'].str.split(", ").str.get(1).str.split(".").str.get(0)

X_temp_title.unique()

#X_temp_last_name = X_temp['Name'].str.split(", ").str.get(0)

X_temp['Name_title'] = X_temp_title

X_temp['Name_len'] = X_temp['Name'].apply(lambda x: len(x))

#X_temp['Last_name'] = X_temp['Name'].apply(lambda x: str.split(x, ",")[0])

X_temp.head(2)
# We further convert name title with low frequency into 'rare'

X_temp['Name_title'] = X_temp['Name_title'].replace(['Mlle','Ms','Mme'], 'Miss')

X_temp['Name_title'] = X_temp['Name_title'].replace(['Lady'], 'Mrs')

title_groups_tmp = X_temp.groupby(['Name_title'])['PassengerId'].count()

title_groups_tmp = title_groups_tmp.apply(lambda x: 'rare' if x < 10 else 'keep')

X_temp['Name_title'] = X_temp['Name_title'].apply(lambda x:x if title_groups_tmp[x] is not 'rare' else 'rare')
X_temp = X_temp.drop(['Name'], axis=1)

X_temp[X_temp['Name_title'] == 'rare'].head()
#fill age missing value with the median grouped by name title

Age_missing_guess = X_temp.groupby(['Name_title'])['Age']

X_temp['Age'] = Age_missing_guess.transform(lambda x: x.fillna(x.median()))



#denote passengers whos age < 16, since children were more likily to be alive.

X_temp['Is_child'] = (X_temp['Age'] <= 13) * 1
# Cut the Age into groups

X_temp['Age_cut'] = pd.cut(X_temp['Age'],5)

X_temp.head()
# Get numbers/zone of Cabin and replace missing values by missing_init

#X_temp['Cabin_len'] = X_temp.Cabin.str.split(' ').apply(lambda x: len(x) if x is not np.nan else missing_init)

#X_temp['Cabin_zone'] = X_temp.Cabin.str[0].apply(lambda x: missing_init if x is np.nan else x)



X_temp['Cabin_isNull'] = X_temp['Cabin'].isna()*1

X_temp['Cabin'] = X_temp.Cabin.fillna('Z')

X_temp.head()
#define fmamily size

X_temp['FamilySize'] = X_temp.SibSp + X_temp.Parch + 1

same_ticket_check = (X_temp.groupby(['Ticket'])['PassengerId'].transform('count') > 1)

family_check = same_ticket_check & (X_temp['FamilySize'] > 1)

friend_check = same_ticket_check & (X_temp['FamilySize'] == 1)

print('Family: {} people'.format((X_temp['FamilySize'] > 1).sum()))

print('Peole who share same ticket: {} peolple'.format(same_ticket_check.sum()))

print('We\'re family - Same ticket and family size>1: {} people'.format(family_check.sum()))

print('We\'re friends - Same ticket and family size=1: {} people'.format(friend_check.sum()))



#create connected survival column

X_temp['connected_survival'] = 0.5

X_temp.connected_survival[:Size_train]





connected_survival_guess = (Y_train.Survived) & (same_ticket_check[:Size_train])*1

X_temp.connected_survival[:Size_train] = connected_survival_guess

X_temp.head()
X_temp.isna().sum()[X_temp.isna().sum() > 0]
#fill na to median for Fare

X_temp['Fare'] = X_temp['Fare'].fillna(X_temp['Fare'].median())

X_temp['Fare'].median()
#fill na to median for Embarked

# Note!! Because there maybe more than 1 mode of one column so df.col.mode() returns a series.

# When padding to missing values with mode, we should code as: df.col.mode()[0] for 1st element of the modes.

X_temp['Embarked'] = X_temp['Embarked'].fillna(X_temp['Embarked'].mode()[0])

X_temp.groupby(['Embarked'])['PassengerId'].count()
# Now the data is clean.

X_temp.isna().sum()
X_temp.head()
X_temp.nunique()
# Some other finetune:



X_temp['Fare_cut'] = pd.cut(X_temp['Fare'],5)

X_temp['Name_len_cut'] = pd.cut(X_temp['Name_len'],5)
X_temp.head()
# normalize Age, Fare, Name_len

X_temp['Age'] = (X_temp['Age'] - X_temp['Age'].min())/(X_temp['Age'].max() - X_temp['Age'].min())

X_temp['Fare'] = (X_temp['Fare'] - X_temp['Fare'].min())/(X_temp['Fare'].max() - X_temp['Fare'].min())

X_temp['Name_len'] = (X_temp['Name_len'] - X_temp['Name_len'].min())/(X_temp['Name_len'].max() - X_temp['Name_len'].min())
# Final adjustment

All_PassengerId = X_temp['PassengerId']

X_temp = X_temp.drop(['Age','PassengerId','SibSp','Parch','Ticket','Name_title','Name_len','Name_len_cut',

                      'Cabin','Embarked'], axis=1)



X_temp.head()
X_temp_oh = pd.get_dummies(X_temp)

X_train_oh = X_temp_oh[:Size_train].copy()

Y_train_oh = Y_train.drop(['PassengerId'], axis=1)

X_test_oh = X_temp_oh[Size_train:].copy()

print('Training set: {} data and {} features after encoding'.format(X_train_oh.shape[0], X_train_oh.shape[1]))

print('Testing set: {} data and {} features after encoding'.format(X_test_oh.shape[0], X_test_oh.shape[1]))
X_temp_oh.head()
Tmodel = Sequential()

#initialize W with normal distribution and b with zeros

# First layer

#Tmodel.add(Dense(input_dim=X_train_oh.shape[1], units=12,

#                 kernel_initializer='normal', bias_initializer='zeros', 

#                 kernel_regularizer=regularizers.l2(0.0001)))

Tmodel.add(Dense(input_dim=X_train_oh.shape[1], units=10))

Tmodel.add(Activation('relu'))



# hidden layers

for i in range(0, 2):

    Tmodel.add(Dropout(0.5))

    #Tmodel.add(Dense(units=8, kernel_initializer='normal', bias_initializer='zeros'))

    Tmodel.add(Dense(units=4))

    Tmodel.add(Activation('relu'))

    

Tmodel.add(Dropout(0.1))

Tmodel.add(Dense(1, activation='sigmoid'))



lr = 0.01

epochs = 7

#decay = lr / epochs

#opt = Adam(lr=lr, beta_1=0.9, beta_2=0.995)

opt = Adam(lr=lr)



Tmodel.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



Tmodel.summary()
stop = EarlyStopping(monitor = 'val_loss', patience = epochs, verbose = 1)



Tmodel.fit(X_train_oh.values, Y_train_oh.values, batch_size=16, epochs=epochs,

           validation_split = 0.25, shuffle = True, verbose=1, callbacks = [stop])



loss, accuracy = Tmodel.evaluate(X_train_oh, Y_train_oh)

print(loss, accuracy)
plt.plot(Tmodel.history.history['acc'])

plt.plot(Tmodel.history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()



plt.plot(Tmodel.history.history['loss'])

plt.plot(Tmodel.history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
Y_pred = Tmodel.predict_classes(X_test_oh.values)
submission = pd.DataFrame()

submission['PassengerId'] = All_PassengerId[Size_train:]

submission['Survived'] = Y_pred

submission.shape
submission.sum()
submission.to_csv('titanic_keras_mlp.csv', index=False)
decisiontree = DecisionTreeClassifier(random_state = 42)
cross_val_score(decisiontree, X_train_oh.values, Y_train_oh.values, cv=30).mean()
decisiontree.fit(X_train_oh.values, Y_train_oh.values)

acc_decision_tree = round(decisiontree.score(X_train_oh.values, Y_train_oh.values), 4)

print("Accuracy: %0.4f" % (acc_decision_tree))
Y_pred_tree = decisiontree.predict(X_test_oh.values)



submission_tree = pd.DataFrame()

submission_tree['PassengerId'] = All_PassengerId[Size_train:]

submission_tree['Survived'] = Y_pred_tree

submission_tree.shape



submission_tree.to_csv('titanic_sk_tree.csv', index=False)
submission_tree.sum()
thresholds = np.linspace(0, 0.5, 50)

# Set the parameters by cross-validation

param_grid = {'min_impurity_split': thresholds, 'max_depth': [6,7,8,9,10]}

 

decisiontree_grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=15, verbose=2)

decisiontree_grid.fit(X_train_oh.values, Y_train_oh.values)

print("best param: {},  best score: {}".format(decisiontree_grid.best_params_, decisiontree_grid.best_score_))



decisiontree_grid.fit(X_train_oh.values, Y_train_oh.values)

acc_decision_tree_grid = round(decisiontree_grid.score(X_train_oh.values, Y_train_oh.values), 4)

print("Accuracy: %0.4f" % (acc_decision_tree_grid))


Y_pred_tree_gs = decisiontree_grid.predict(X_test_oh.values)



submission_tree_gs = pd.DataFrame()

submission_tree_gs['PassengerId'] = All_PassengerId[Size_train:]

submission_tree_gs['Survived'] = Y_pred_tree_gs

submission_tree_gs.shape
submission_tree_gs.sum()
submission_tree_gs.to_csv('titanic_sk_tree_gs.csv', index=False)
random_forest = RandomForestClassifier(random_state = 40, n_estimators=500, oob_score=False, max_depth=4,

                                      max_leaf_nodes=7, min_impurity_split=0.001, max_features='log2',

                                      min_samples_leaf=3,

                                      min_impurity_decrease=0.00000001,

                                      min_weight_fraction_leaf=0.06, warm_start=True,

                                      class_weight={0: 1, 1: 1.3})

random_forest.fit(X_train_oh.values, Y_train_oh.values)
#cross_val_score(random_forest, X_train_oh.values, Y_train_oh.values, cv=10).mean()
acc_decision_rf = round(random_forest.score(X_train_oh.values, Y_train_oh.values), 4)

print("Accuracy: %0.4f" % (acc_decision_rf))
Y_pred_rf = random_forest.predict(X_test_oh.values)



submission_rf = pd.DataFrame()

submission_rf['PassengerId'] = All_PassengerId[Size_train:]

submission_rf['Survived'] = Y_pred_rf





submission_rf.shape
submission_rf.sum()
submission_rf.to_csv('titanic_sk_rf.csv', index=False)
### No use in this notebook finally, just for reference.

# Build multi-columns label encoder by using LabelEncoder in sklearn

if 1 != 1:

    class Multi_Column_LabelEncoder:

        def __init__(self, columns=None):

            self.columns = columns



        def fit(self, X, y=None):

            return self



        def transform(self, X):

            '''

            Transforms columns of X specified in self.columns using

            LabelEncoder(). If no columns specified, transforms all

            columns in X.

            '''

            output = X.copy()



            # Because we've fill na with 0, some columns is mix-typed. 

            # LE need the column value same type so we fix it with: output[col_name].astype(str)

            if self.columns is None:

                for col_name in output.columns:

                    output[col_name] = LabelEncoder().fit_transform(output[col_name].astype(str))

            else:

                for col_name in self.columns:

                    output[col_name] = LabelEncoder().fit_transform(output[col_name].astype(str))

            return output



        def fit_transform(self, X, y=None):

            return self.fit(X,y).transform(X)

        

        

####################################

# Encoding and pre-modeling

####################################                  

'''

# dropping useless features

data = data.drop(columns = ['Age','Cabin','Embarked','Name','Last_Name',

                            'Parch', 'SibSp','Ticket', 'Family_Size'])



# Encoding features

target_col = ["Survived"]

id_dataset = ["Type"]

cat_cols   = data.nunique()[data.nunique() < 12].keys().tolist()

cat_cols   = [x for x in cat_cols ]

# numerical columns

num_cols   = [x for x in data.columns if x not in cat_cols + target_col + id_dataset]

# Binary columns with 2 values

bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()

# Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]

# Label encoding Binary columns

le = LabelEncoder()

for i in bin_cols :

    data[i] = le.fit_transform(data[i])

# Duplicating columns for multi value columns

data = pd.get_dummies(data = data,columns = multi_cols )

# Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(data[num_cols])

scaled = pd.DataFrame(scaled,columns = num_cols)

# dropping original values merging scaled values for numerical columns

df_data_og = data.copy()

data = data.drop(columns = num_cols,axis = 1)

data = data.merge(scaled,left_index = True,right_index = True,how = "left")

data = data.drop(columns = ['PassengerId'],axis = 1)



# Target = 1st column

cols = data.columns.tolist()

cols.insert(0, cols.pop(cols.index('Survived')))

data = data.reindex(columns= cols)



# Cutting train and test

train = data[data['Type'] == 1].drop(columns = ['Type'])

test = data[data['Type'] == 0].drop(columns = ['Type'])



# X and Y

X_train = train.iloc[:, 1:20].as_matrix()

y_train = train.iloc[:,0].as_matrix()

''' 




