#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#Make data frames
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#Training data frame with 143 features
train.head()
#Testing without target feature
test.head()
#See what values are missing >%50
train.isnull().sum().sort_values(ascending=False)
#Droped the features that have and missing values and/or were redundant
train.drop(labels=(['v2a1', 'v18q1', 'rez_esc', 'tamviv', 'r4h3', 'r4h3', 'r4t1', 'r4t2', 'r4t3', 'meaneduc', 'SQBescolari',
                   'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 
                   'SQBmeaned', 'agesq']), axis=1, inplace=True)
#Sanity check #001
train.shape
#Get the value counts of our target variable
train.Target.value_counts()
#Some visualization for the target variable
plt.hist(train.Target, edgecolor='black')
plt.xticks([1, 2, 3, 4])
plt.xlabel("Poverty Level")
plt.ylabel("Value Counts")
plt.title("Value Counts for Poverty in Costa Rica")
#Create a categorical version of the target column
mapped = {1: 'extreme poverty', 2:'moderate poverty', 3: 'vulnerable', 4: 'non_vulnerable'}
train['Target_cat'] = train['Target']
train.Target_cat = train.Target_cat.replace(mapped)
#sanity check #002
train.Target_cat.value_counts()
#Create a graph to show the levels of poverty
train.Target_cat.value_counts().plot.barh(color='blue', edgecolor='black')
plt.ylabel('Poverty level')
plt.xlabel('Number of People')
plt.title('Poverty levels by person')
print("Non-vulnerable cases make up", round((train.Target_cat.value_counts()[0] / train.shape[0]) * 100, 2), 
      "% of all cases.")
#Replace numbers with each person's gender
gender_map = {1: 'Female', 0: 'Male'}
train['Gender'] = train.female.replace(gender_map)
train.drop(['male'], axis=1, inplace=True)
#See the beakdown of gender when it comes to poverty
print(train.groupby(['Target_cat']).Gender.value_counts())
train.groupby(['Target_cat']).Gender.value_counts().plot.barh(color='blue', edgecolor='black')
plt.ylabel('Poverty Level by Gender')
plt.xlabel('Number of people')
plt.title('Number of people in poverty by gender')
#See the breakdown of poverty by whether someone has a disability
dis_map = {1: 'Handicapped', 0: 'Non_Handicapped'}
train['Disability'] = train.dis.replace(dis_map)
train.drop('dis', axis=1, inplace=True)
print(train.groupby(['Target_cat']).Disability.value_counts())
train.groupby(['Target_cat']).Disability.value_counts().plot.barh(color='blue', edgecolor='black')
#Hmmmm it was hard to tell from that graph. Let's try by percentages
dis_counts = train.groupby(['Target_cat']).Disability.value_counts()
print(dis_counts)
dis_counts = [round(53 / (702 + 53) * 100, 2), round(135 / (1462 + 135) * 100, 2), round(285 / (5711 + 285) * 100, 2), round(77 / (1132 + 77) * 100, 2)]
print("Percentage of impoverished by disability:", dis_counts)
#Get a list of feature types
train.info(verbose=True)
#See what the 8 object categories are
train.select_dtypes('object').head(10)
#Replace yes with 1 and no with 0
map_resp = {'yes': 1, 'no': 0}
cats = ['dependency', 'edjefe', 'edjefa']
for x in cats:
    train[x] = train[x].replace(map_resp)
    test[x] = test[x].replace(map_resp)
#Sanity check #003
train.select_dtypes('object').head(10)
#Let's check how many of these columns are Boolean
print(train.select_dtypes('int64').nunique().value_counts().sort_index())
train.select_dtypes('int64').nunique().value_counts().sort_index().plot.bar(color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Number of features with unique value')
plt.title('Number of Unique Values per feature')
#Checking the column with only one unique value
One_val = []
for x in train.columns:
    if train[x].nunique() == 1:
        One_val.append(x)
    else:
        pass
train_one = train[One_val]
train_one.head()
#Sanity check #004
train_one.elimbasu5.value_counts()
#Alright let's drop it
train = train.drop(['elimbasu5'], axis=1)
train.shape
#A seperate dataframe is created to analyize the boolean columns
bool_col = []
for x in train.columns:
    if train[x].nunique() == 2:
        bool_col.append(x)
    else:
        pass
train_bool = train[bool_col]
train_bool.head()
#Create sub dataframes depending on each level of poverty
train_bool['Target'] = train['Target']
extreme_mask = train_bool.Target == 1
extreme = train_bool[extreme_mask]
moderate_mask = train_bool.Target == 2
moderate = train_bool[moderate_mask]
vunlnerable_mask = train_bool.Target == 3
vunlnerable = train_bool[vunlnerable_mask]
non_vunlnerable_mask = train_bool.Target == 4
non_vunlnerable = train_bool[non_vunlnerable_mask]
#note: I'm aware I spelled vulnerable wrong
#See how many cases are in each level
print(extreme.shape)
print(moderate.shape)
print(vunlnerable.shape)
print(non_vunlnerable.shape)
#Generate a report of how much each column makes up when it comes to extreme poverty
name = []
mean = []
std = []
train_bool = train_bool.drop(['Target'], axis=1)
for x in train_bool.columns:
    print(x + ":")
    mask = extreme[x] == 1
    extreme1 = extreme[mask]
    print(x, "makes up", round(extreme1.shape[0] / extreme.shape[0] * 100, 2), "% of the extreme category.")
    mask = moderate[x] == 1
    moderate1 = moderate[mask]
    print(x, "makes up", round(moderate1.shape[0] / moderate.shape[0] * 100, 2), "% of the moderate category.")
    mask = vunlnerable[x] == 1
    vunlnerable1 = vunlnerable[mask]
    print(x, "makes up", round(vunlnerable1.shape[0] / vunlnerable.shape[0] * 100, 2), "% of the vunlnerable category.")
    mask = non_vunlnerable[x] == 1
    non_vunlnerable1 = non_vunlnerable[mask]
    print(x, "makes up", round(non_vunlnerable1.shape[0] / non_vunlnerable.shape[0] * 100, 2), "% of the non_vunlnerable category.")
    combined = np.array([extreme1.shape[0] / extreme.shape[0] * 100, moderate1.shape[0] / moderate.shape[0] * 100, vunlnerable1.shape[0] / vunlnerable.shape[0] * 100, non_vunlnerable1.shape[0] / non_vunlnerable.shape[0] * 100])
    print(x, "Mean:", round(combined.mean(), 2), "%")
    print(x, "Std:", round(combined.std(), 2), "%")
    name.append(x)
    mean.append(combined.mean())
    std.append(combined.std())
    print("\n")
#Generate a data frame to determine priority of columns
name = np.array(name)
mean = np.array(mean)
std = np.array(std)
Boolean_df = pd.DataFrame({'Column': name, 'Mean': mean, 'Standard Deviation': std})
Boolean_df.sort_values(by='Standard Deviation', ascending=False).reset_index().head(10)

#Generate a data frame for the rest of the columns
int_col = []
for x in train.columns:
    if train[x].nunique() > 2:
        int_col.append(x)
    else:
        pass
train_int = train[int_col]
train_int.head()
#We should drop the Id column from the data frame. Let's check what else needs to be dropped
train_int.columns
#Looks like we can drop Id and idhogar as these are identifiers. Let's also drop hogar total since it's redundent with
#the other breakdowns by day. Let's also drop hogar_total from the main data frame as well
train = train.drop(['hogar_total'], axis=1)
#Let's see which featuers would be important in a model
x_list = train.columns
x_list = x_list.drop(['Id', 'Target', 'Target_cat', 'idhogar'])
X = train[x_list]
X = X.drop(['Disability', 'Gender'], axis=1)
y = train.Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Let's see how the model scores
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
#A Confusion Matrix shows where the inaccurate predictions are
values = rf.predict(X_test)
print(np.unique(values, return_counts=True))
confusion_matrix(y_test, values)
#See the importance of the features
features = rf.feature_importances_
summation = []
var_name = []
for feature in zip(x_list, features):
    if feature[1] > .01:
        print(feature)
        summation.append(feature[1])
        var_name.append(feature[0])
    else:
        pass
summation = np.array(summation)
print(str(len(summation)) + " variables account for " + str(round(summation.sum() * 100, 2)) + "% of the variation")
#Use train as the training data and test as the test data
X_train = train[var_name]
y_train = train['Target']
X_test = test[var_name]
#Fit the random forest and predict
rf.fit(X_train, y_train)
y_test = rf.predict(X_test)
test['Target'] = y_test
#sanity check #005
test.Target.value_counts()
#Submission process
sub_var = ['Id', 'Target']
submission = test[sub_var]
submission.to_csv('Submission.csv', index=False)