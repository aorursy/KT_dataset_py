import pandas as pd
df = pd.read_csv('../input/environmental-sensor-data-132k/iot_telemetry_data.csv')
df.head()
df['hour'] = pd.to_datetime(df['ts'],unit='s').dt.hour

df['minute'] = pd.to_datetime(df['ts'],unit='s').dt.minute

df['second'] = pd.to_datetime(df['ts'],unit='s').dt.second

df['microsecond'] = pd.to_datetime(df['ts'],unit='s').dt.microsecond
df = df.drop('ts', axis = 1)
df.device.unique()
codes, uniques = df.device.factorize()

print(codes)

print(uniques)
df['deviceFactor'] = codes
df = df.drop('device', axis = 1)
from sklearn.model_selection import train_test_split
df[df.motion == True].motion.count()
df[df.motion == False].motion.count()
df.motion.count()
df_true_train, df_true_test = train_test_split(df[df.motion == 1], test_size = 0.25, random_state = 42)
df_true_train.motion.count()
df_false_big, df_false_lit = train_test_split(df[df.motion == 0], test_size = 0.00119, random_state = 42)
df_false_lit.motion.count()
df_false_train, df_false_test = train_test_split(df_false_lit, test_size = 0.25, random_state = 42)
df_false_train.motion.count()
df_false_test.motion.count()
result_train = pd.concat([df_true_train, df_false_train])
result_train.motion.count()
result_train[result_train.motion == True].motion.count()
result_test = pd.concat([df_true_test, df_false_test])
result_test.motion.count()
result_test[result_test.motion == True].motion.count()
X_train = result_train.drop('motion', axis = 1)

y_train = result_train.motion
X_test = result_test.drop('motion', axis = 1)

y_test = result_test.motion
X = df.drop('motion', axis = 1)

y = df.motion
df.head()
print('Decision Tree Classifier:')

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state = 42)

tree.fit(X_train, y_train)

print('train score: ' + str(tree.score(X_train, y_train)))

print('test score: ' + str(tree.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix

y_pred = tree.predict(X_test)

print('confusion matrix:\n' + str(matrx))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y_test, y_pred)))

matrx = confusion_matrix(y_test, y_pred)

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))

print()

from sklearn.metrics import confusion_matrix

y_pred = tree.predict(X)

print('On full dataframe:')

from sklearn.metrics import accuracy_score

print('accuracy: ' + str(accuracy_score(y, y_pred)))

matrx = confusion_matrix(y, y_pred)

print('confusion matrix:\n' + str(matrx))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y, y_pred)))

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))
print('Random Forest Classifier:')

from  sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state = 42, n_estimators = 250)

forest.fit(X_train, y_train)

print('train score: ' + str(forest.score(X_train, y_train)))

print('test score: ' + str(forest.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix

y_pred = forest.predict(X_test)

print('confusion matrix:\n' + str(confusion_matrix(y_test, y_pred)))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y_test, y_pred)))

matrx = confusion_matrix(y_test, y_pred)

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))

print()

from sklearn.metrics import confusion_matrix

y_pred = forest.predict(X)

matrx = confusion_matrix(y, y_pred)

print('On full dataframe:')

from sklearn.metrics import accuracy_score

print('accuracy: ' + str(accuracy_score(y, y_pred)))

print('confusion matrix:\n' + str(matrx))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y, y_pred)))

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))
print('Gradient Boosting Classifier:')

from sklearn.ensemble import GradientBoostingClassifier

boost = GradientBoostingClassifier(random_state = 42, n_estimators = 500, learning_rate = 0.1)

boost.fit(X_train, y_train)

print('train score: ' + str(boost.score(X_train, y_train)))

print('test score: ' + str(boost.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix

y_pred = boost.predict(X_test)

print('confusion matrix:\n' + str(confusion_matrix(y_test, y_pred)))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y_test, y_pred)))

matrx = confusion_matrix(y_test, y_pred)

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))

print()

from sklearn.metrics import confusion_matrix

y_pred = boost.predict(X)

print('On full dataframe:')

from sklearn.metrics import accuracy_score

print('accuracy: ' + str(accuracy_score(y, y_pred)))

print('confusion matrix:\n' + str(matrx))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y, y_pred)))

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))
%%time

print('Time spended by tree')

y_pred = tree.predict(X)

matrx = confusion_matrix(y, y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y, y_pred))

print(matrx)

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y, y_pred))
%%time

print('Time spended by forest')

y_pred = forest.predict(X)

matrx = confusion_matrix(y, y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y, y_pred))

print(matrx)

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y, y_pred))
%%time

print('Time spended by gradient boosting classifier')

y_pred = boost.predict(X)

matrx = confusion_matrix(y, y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y, y_pred))

print(matrx)

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y, y_pred))
print('LogisticRegression:')

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

lg.fit(X_train, y_train)

print('train set score: ', end='')

print(lg.score(X_train, y_train))

print('test set score: ', end='')

print(lg.score(X_test, y_test))

print('full dataframe score: ', end='')

print(lg.score(X, y))

from sklearn.metrics import confusion_matrix

y_pred = lg.predict(X)

matrx = confusion_matrix(y, y_pred)

print('confusion matrix:')

print(matrx)

from sklearn.metrics import roc_auc_score

print('roc auc score: ', end='')

print(roc_auc_score(y, y_pred))

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))
print('SGDClassifier:')

from sklearn.linear_model import SGDClassifier

SGD = SGDClassifier()

SGD.fit(X_train, y_train)

print('train set score: ', end='')

print(SGD.score(X_train, y_train))

print('test set score: ', end='')

print(SGD.score(X_test, y_test))

print('full dataframe score: ', end='')

print(SGD.score(X, y))

from sklearn.metrics import confusion_matrix

y_pred = SGD.predict(X)

matrx = confusion_matrix(y, y_pred)

print('confusion matrix:')

print(matrx)

from sklearn.metrics import roc_auc_score

print('roc auc score: ', end='')

print(roc_auc_score(y, y_pred))

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))
print('GaussianNB:')

from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()

GNB.fit(X_train, y_train)

print('train set score: ', end='')

print(GNB.score(X_train, y_train))

print('test set score: ', end='')

print(GNB.score(X_test, y_test))

print('full dataframe score: ', end='')

print(GNB.score(X, y))

from sklearn.metrics import confusion_matrix

y_pred = GNB.predict(X)

matrx = confusion_matrix(y, y_pred)

print('confusion matrix:')

print(matrx)

from sklearn.metrics import roc_auc_score

print('roc auc score: ', end='')

print(roc_auc_score(y, y_pred))

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))
df_true_train, df_true_test = train_test_split(df[df.motion == 1], test_size = 0.25, random_state = 42)
df_false_big, df_false_lit = train_test_split(df[df.motion == 0], test_size = 0.01, random_state = 42)
df_false_lit.motion.count()
df_false_train, df_false_test = train_test_split(df_false_lit, test_size = 0.25, random_state = 42)
result_train = pd.concat([df_true_train, df_false_train])
result_test = pd.concat([df_true_test, df_false_test])
X_train = result_train.drop('motion', axis = 1)

y_train = result_train.motion
X_test = result_test.drop('motion', axis = 1)

y_test = result_test.motion
X = df.drop('motion', axis = 1)

y = df.motion
print('Decision Tree Classifier:')

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state = 42)

tree.fit(X_train, y_train)

print('train score: ' + str(tree.score(X_train, y_train)))

print('test score: ' + str(tree.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix

y_pred = tree.predict(X_test)

print('confusion matrix:\n' + str(confusion_matrix(y_test, y_pred)))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y_test, y_pred)))

matrx = confusion_matrix(y_test, y_pred)

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))

print()

from sklearn.metrics import confusion_matrix

y_pred = tree.predict(X)

matrx = confusion_matrix(y, y_pred)

print('On full dataframe:')

from sklearn.metrics import accuracy_score

print('accuracy: ' + str(accuracy_score(y, y_pred)))

print('confusion matrix:\n' + str(matrx))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y, y_pred)))

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))
print('Random Forest Classifier:')

from  sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state = 42, n_estimators = 750)

forest.fit(X_train, y_train)

print('train score: ' + str(forest.score(X_train, y_train)))

print('test score: ' + str(forest.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix

y_pred = forest.predict(X_test)

print('confusion matrix:\n' + str(confusion_matrix(y_test, y_pred)))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y_test, y_pred)))

matrx = confusion_matrix(y_test, y_pred)

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))

print()

from sklearn.metrics import confusion_matrix

y_pred = forest.predict(X)

matrx = confusion_matrix(y, y_pred)

print('On full dataframe:')

from sklearn.metrics import accuracy_score

print('accuracy: ' + str(accuracy_score(y, y_pred)))

print('confusion matrix:\n' + str(matrx))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y, y_pred)))

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))
print('Gradient Boosting Classifier:')

from sklearn.ensemble import GradientBoostingClassifier

boost = GradientBoostingClassifier(random_state = 42, n_estimators = 750, learning_rate = 0.2)

boost.fit(X_train, y_train)

print('train score: ' + str(boost.score(X_train, y_train)))

print('test score: ' + str(boost.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix

y_pred = boost.predict(X_test)

print('confusion matrix:\n' + str(confusion_matrix(y_test, y_pred)))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y_test, y_pred)))

matrx = confusion_matrix(y_test, y_pred)

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))

print()

from sklearn.metrics import confusion_matrix

y_pred = boost.predict(X)

matrx = confusion_matrix(y, y_pred)

print('On full dataframe:')

from sklearn.metrics import accuracy_score

print('accuracy: ' + str(accuracy_score(y, y_pred)))

print('confusion matrix:\n' + str(matrx))

from sklearn.metrics import roc_auc_score

print('roc_auc: ' + str(roc_auc_score(y, y_pred)))

print('recall for negative samples(motion = False): ' + str(matrx[1][1] / (matrx[1][0] + matrx[1][1])))