import pandas as pd
import numpy as np
from sklearn import preprocessing
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#import pydotplus
# csv file -> dataframe
df = pd.read_csv('../input/wineData.csv')
df.head(10)
df.describe()
# class map target features
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['Class']))}
tempCol = df.pop('Class')
# normalized dataframe
min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 2))
df_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
df_scaled['Class'] = tempCol
df_scaled['Class'] = df_scaled['Class'].map(class_mapping)
pprint(df_scaled)
# dataframe -> csv file
df_scaled.to_csv('wineNormalized.csv', index=False)
# csv file -> dataframe
df = pd.read_csv('wineNormalized.csv')
# Target Feature -> Y & descriptive features -> X
x, y = df.iloc[:, :-1].values, df['Class'].values
# dataframe -> training/testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.33, random_state=0)

df.pop('Class')
# numpy arrays -> dataframes
df_train = pd.DataFrame(x_train, columns=df.columns)
df_train['Class'] = y_train
df_test = pd.DataFrame(x_test, columns=df.columns)
df_test['Class'] = y_test
# datasets -> csv files
df_train.to_csv('train_dataset.csv', index=False)
df_test.to_csv('test_dataset.csv', index=False)
resultsEntropy = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=0)
    clf = clf.fit(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    train_score = clf.score(x_train, y_train)
    resultsEntropy.loc[i] = [i, train_score, test_score]
    
resultsEntropy.pop('LevelLimit')
resultsEntropy.plot()
plt.title('Entropy Decision Tree Score Chart')
plt.xlabel("tree depth level")
plt.ylabel("% accuracy")
plt.show()
resultsGini = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=0)
    clf = clf.fit(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    train_score = clf.score(x_train, y_train)
    resultsGini.loc[i] = [i, train_score, test_score]
   
resultsGini.pop('LevelLimit')
resultsGini.plot()
plt.title('Gini Decision Tree Score Chart')
plt.xlabel("tree depth level")
plt.ylabel("% accuracy")
plt.show()
clf = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=0)
clf = clf.fit(x_train, y_train)

dot_data = export_graphviz(clf,
                           feature_names=df.columns,
                           out_file=None,
                           filled=True,
                           rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png('tree.png')
# Loading the csv file in a dataframe
df_train = pd.read_csv('train_dataset.csv')
df_test = pd.read_csv('test_dataset.csv')

# dataframe -> training and test datasets
x_train, y_train = df_train.iloc[:, :-1].values, df_train['Class'].values
x_test, y_test = df_test.iloc[:, :-1].values, df_test['Class'].values

# euclidean - uniform (e1)
k_e1 = pd.DataFrame(columns=['K neighbors', 'Training Score', 'Testing Score'])
avg_train_e1 = []
avg_test_e1 = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i, weights='uniform', p=2, metric='minkowski')
    knn.fit(x_train, y_train)

    test_score = knn.score(x_test, y_test)
    train_score = knn.score(x_train, y_train)

    k_e1.loc[i] = [i, train_score, test_score]

    avg_train_e1.append(train_score)
    avg_test_e1.append(test_score)

avg_trainscore_e1 = sum(avg_train_e1)/len(avg_train_e1)
print('average train score for e1 =', avg_trainscore_e1)

avg_testscore_e1 = sum(avg_test_e1)/len(avg_test_e1)
print('average test score for e1 =', avg_testscore_e1)


# euclidean - weighted distance (e2)
k_e2 = pd.DataFrame(columns=['K neighbors', 'Training Score', 'Testing Score'])
x_e2 = []
y_e2 = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', p=2, metric='minkowski')
    knn.fit(x_train, y_train)

    test_score = knn.score(x_test, y_test)
    train_score = knn.score(x_train, y_train)

    k_e2.loc[i] = [i, train_score, test_score]
    x_e2.append(test_score)
    y_e2.append(train_score)

mean_y_e2 = sum(y_e2)/len(y_e2)
print('mean training test score for e2 is', mean_y_e2)

mean_e2 = sum(x_e2)/len(x_e2)
print(mean_e2)

# manhattan - uniform
k_m1 = pd.DataFrame(columns=['K neighbors', 'Training Score', 'Testing Score'])
x_m1 = []
y_m1 = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i, weights='uniform', p=1, metric='minkowski')
    knn.fit(x_train, y_train)

    test_score = knn.score(x_test, y_test)
    train_score = knn.score(x_train, y_train)

    k_m1.loc[i] = [i, train_score, test_score]
    x_m1.append(test_score)
    y_m1.append(train_score)

mean_y_m1 = sum(y_m1)/len(y_m1)
print('m1', mean_y_m1)

mean_m1 = sum(x_m1)/len(x_m1)
print(mean_m1)


# manhattan - weighted distance
k_m2 = pd.DataFrame(columns=['K neighbors', 'Training Score', 'Testing Score'])
x_m2 = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', p=1, metric='minkowski')
    knn.fit(x_train, y_train)

    test_score = knn.score(x_test, y_test)
    train_score = knn.score(x_train, y_train)

    k_m2.loc[i] = [i, train_score, test_score]
    x_m2.append(test_score)

mean_m2 = sum(x_m2)/len(x_m2)
print(mean_m2)


# plotting
# euclidean - uniform
k_e1.pop('K neighbors')
k_e1.plot()
plt.title('KNN Euclidean Uniform Score Chart')
plt.ylabel('% Accuracy')
plt.xlabel('k value')
plt.show()

# euclidean - weighted distance
k_e2.pop('K neighbors')
k_e2.plot()
plt.title('KNN Euclidean Weighted-Distance Score Chart')
plt.ylabel('% Accuracy')
plt.xlabel('k value')
plt.show()

# manhattan - uniform
k_m1.pop('K neighbors')
k_m1.plot()
plt.title('KNN Manhattan Uniform Score Chart')
plt.ylabel('% Accuracy')
plt.xlabel('k value')
plt.show()

# manhattan - weighted distance
k_m2.pop('K neighbors')
k_m2.plot()
plt.title('KNN Manhattan Weighted-Distance Score Chart')
plt.ylabel('% Accuracy')
plt.xlabel('k value')
plt.show()
# Loading the csv file in a dataframe
df_train = pd.read_csv('train_dataset.csv')
df_test = pd.read_csv('test_dataset.csv')



top_10_features = [ 'Color intensity', 'Flavanoids', 'Alcohol','Total phenols',] #'Proline','Magnesium', 'Nonflavanoid phenols']#, 'Malic acid', 'Ash', 'OD280/OD315 of diluted wines','Alcalinity of ash', 'Hue']



df_train_topfeatures = df_train[top_10_features]
df_test_topfeatures = df_test[top_10_features]

df_train_topfeatures['Proline + Magnesium'] = ((df_train['Proline'] + df_train['Magnesium'])/2)
df_test_topfeatures['Proline + Magnesium'] = ((df_test['Proline'] + df_test['Magnesium'])/2)

x_train, y_train = df_train_topfeatures.iloc[:, :].values, df_train['Class'].values
x_test, y_test = df_test_topfeatures.iloc[:, :].values, df_test['Class'].values

scores = {}
results = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=0)
    clf = clf.fit(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    train_score = clf.score(x_train, y_train)
    results.loc[i] = [i, train_score, test_score]

results.pop('LevelLimit')
results.plot()
plt.title('Decision Tree with Select Features + Derived Feature')
plt.xlabel('Max Tree Depth Level')
plt.ylabel('% Accuracy')
plt.show()

df_train['Proline + Magnesium'] = ((df_train['Proline'] + df_train['Magnesium'])/2)
df_test['Proline + Magnesium'] = ((df_test['Proline'] + df_test['Magnesium'])/2)

y_train_temp = df_train.pop('Class')
y_test_temp = df_test.pop('Class')

x_train, y_train = df_train.iloc[:, :].values, y_train_temp.values
x_test, y_test = df_test.iloc[:, :].values, y_test_temp.values

resultsDerived = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=0)
    clf = clf.fit(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    train_score = clf.score(x_train, y_train)
    resultsDerived.loc[i] = [i, train_score, test_score]

resultsDerived.pop('LevelLimit')
resultsDerived.plot()
plt.title('Decision Tree Results from Derived Features')
plt.show()

# Joining Top 10 Features with the new Derived ones
df_train_latest = df_train_derived.join(df_train_topfeatures)
df_test_latest = df_test_derived.join(df_train_topfeatures)

X_train, y_train = df_train_latest.iloc[:, :].values, df_train['Class'].values
X_test, y_test = df_test_latest.iloc[:, :].values, df_test['Class'].values

resultsMixed = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=0)
    clf = clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    train_score = clf.score(X_train, y_train)
    resultsMixed.loc[i] = [i, train_score, test_score]

resultsMixed.pop('LevelLimit')
resultsMixed.plot()
plt.title('Decision Tree Results from Derived and Top Features')
plt.show()
# The link to the Final Report and conclusions are in the Github link below
# https://github.com/vamsivchinta1/Wine-Classification-Project.git