import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
trainSet = pd.read_csv("../input/train.csv")
testSet = pd.read_csv("../input/test.csv")

trainSet.columns
trainSet.head()
print(trainSet.dtypes)
print(trainSet.describe())
Survived_m = trainSet.Survived[trainSet.Sex == 'male'].value_counts()
Survived_f = trainSet.Survived[trainSet.Sex == 'female'].value_counts()

print(Survived_m)
print(Survived_f)

df = pd.DataFrame({'male': Survived_m, 'female': Survived_f})
df.plot(kind = 'bar', stacked = True)
plt.title('Survived by sex')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
trainSet['Age'].hist()
plt.ylabel('Count')
plt.xlabel('Age')
plt.title('Survived by Age')
plt.show()

trainSet[trainSet.Survived == 0].Age.hist()
plt.ylabel('Count')
plt.xlabel('Age')
plt.title('Not Survived by Age')
plt.show()

trainSet[trainSet.Survived == 1].Age.hist()
plt.ylabel('Count')
plt.xlabel('Age')
plt.title('Survived by Age')
plt.show()
trainSet.Fare.hist()
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution')
plt.show() 

trainSet[trainSet.Survived == 0].Fare.hist()
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Not Survived Fare distribution')
plt.show() 

trainSet[trainSet.Survived == 1].Fare.hist()
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Survived Fare distribution')
plt.show() 
trainSet['Pclass'].hist()
plt.show()
print(trainSet['Pclass'].isnull().values.any())

Survived_p1 = trainSet.Survived[trainSet['Pclass'] == 1].value_counts()
Survived_p2 = trainSet.Survived[trainSet['Pclass'] == 2].value_counts()
Survived_p3 = trainSet.Survived[trainSet['Pclass'] == 3].value_counts()

df = pd.DataFrame({'p1': Survived_p1, 'p2': Survived_p2, 'p3': Survived_p3})
print(df)
df.plot(kind = 'bar', stacked = True)
plt.title('Survived by Pclass')
plt.ylabel('Count')
plt.xlabel('Survival by Pclass')
plt.show()
Survived_S = trainSet.Survived[trainSet.Embarked == 'S'].value_counts()
Survived_C = trainSet.Survived[trainSet.Embarked == 'C'].value_counts()
Survived_Q = trainSet.Survived[trainSet.Embarked == 'Q'].value_counts()

print(Survived_S)
df_Embarked = pd.DataFrame({'S': Survived_S, 'C': Survived_C, 'Q': Survived_Q})
df_Embarked.plot(kind = 'bar', stacked = True)
plt.title('Survival by Embarked')
plt.ylabel('Count')
plt.xlabel('Survival by Embarked')
plt.show()
trainLabel = trainSet.Survived
trainData = trainSet.loc[:, ['Sex','Age','Fare','Pclass','Embarked']]
testData = testSet.loc[:, ['Sex','Age','Fare','Pclass','Embarked']]
print(trainData)
print(testData)
def fill_NA(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[:, 'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:, 'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:, 'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:, 'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:, 'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

trainData_no_nan = fill_NA(trainData)
testData_no_nan = fill_NA(testData)
print(trainData_no_nan)
print(testData_no_nan)
def transfer_Sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    return data_copy

trainData_after_sex = transfer_Sex(trainData_no_nan)
testData_after_sex = transfer_Sex(testData_no_nan)
print(testData_after_sex)
def transfer_Embark(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

trainData_after_embark = transfer_Embark(trainData_after_sex)
testData_after_embark = transfer_Embark(testData_after_sex)
print(testData_after_embark)
data_now = trainData_after_embark
testData_now = testData_after_embark
from sklearn.model_selection import train_test_split

train_data1, test_data1, train_labels1, test_labels1 = train_test_split(data_now, trainLabel, test_size = 0.2, random_state = 0)
print(train_data1.shape, test_data1.shape, train_labels1.shape, test_labels1.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

k_range = range(1, 51)
k_accuracy_scores = []
k_precision_scores = []
k_recall_scores = []
k_f1_scores = []
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    clf.fit(train_data1, train_labels1)
    predictions = clf.predict(test_data1)
    
    accuracyScore = accuracy_score(test_labels1, predictions)
    k_accuracy_scores.append(accuracyScore)
    
    precisionScore = precision_score(test_labels1, predictions)
    k_precision_scores.append(precisionScore)
    
    recallScore = recall_score(test_labels1, predictions)
    k_recall_scores.append(recallScore)
    
    f1Score = f1_score(test_labels1, predictions)
    k_f1_scores.append(f1Score)
# 检测模型precision， recall 等各项指标
plt.plot(k_range, k_accuracy_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()

plt.plot(k_range, k_precision_scores)
plt.xlabel('K for KNN')
plt.ylabel('Precision on validation set')
plt.show()

plt.plot(k_range, k_recall_scores)
plt.xlabel('K for KNN')
plt.ylabel('Recall on validation set')
plt.show()

plt.plot(k_range, k_recall_scores)
plt.xlabel('K for KNN')
plt.ylabel('f1 on validation set')
plt.show()

print('For first run (before running K folds)')

sorted_K_by_accuracy = np.array(k_accuracy_scores).argsort()
#print(sorted_K_by_accuracy)
best_K_by_accuracy_1 = sorted_K_by_accuracy[len(sorted_K_by_accuracy)-1] + 1
print('Best K by accuracy is ', best_K_by_accuracy_1)

sorted_K_by_precision = np.array(k_precision_scores).argsort()
#print(sorted_K_by_precision)
best_K_by_precision_1 = sorted_K_by_precision[len(sorted_K_by_precision)-1] + 1
print('Best K by precision is ',best_K_by_precision_1)

sorted_K_by_recall = np.array(k_recall_scores).argsort()
#print(sorted_K_by_recall)
best_K_by_recall_1 = sorted_K_by_recall[len(sorted_K_by_recall)-1] + 1
print('Best K by recall is ',best_K_by_recall_1)

sorted_K_by_f1 = np.array(k_f1_scores).argsort()
#print(sorted_K_by_f1)
best_K_by_f1_1 = sorted_K_by_f1[len(sorted_K_by_f1)-1] + 1
print('Best K by f1 is ',best_K_by_f1_1)
# cross validation 找到最好的k值

from sklearn.model_selection import cross_val_score
score_list_accuracy = []
score_list_precision = []
score_list_recall = []
score_list_f1 = []
for K in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors = K)
    score_list_accuracy.append(cross_val_score(clf, data_now, trainLabel, scoring='accuracy', cv = 5).mean())
    score_list_precision.append(cross_val_score(clf, data_now, trainLabel, scoring='precision', cv = 5).mean())
    score_list_recall.append(cross_val_score(clf, data_now, trainLabel, scoring='recall', cv = 5).mean())
    score_list_f1.append(cross_val_score(clf, data_now, trainLabel, scoring='f1', cv = 5).mean())
    
#print('Accuracy score list is', score_list_accuracy)
sorted_cross_validated_k_accuracy = np.array(score_list_accuracy).argsort()
sorted_cross_validated_k_precision = np.array(score_list_precision).argsort()
sorted_cross_validated_k_recall = np.array(score_list_recall).argsort()
sorted_cross_validated_k_f1 = np.array(score_list_f1).argsort()

#print(sorted_cross_validated_k_by_accuracy)
best_K_cross_validated_accuracy = sorted_cross_validated_k_accuracy[len(sorted_cross_validated_k_accuracy)-1] + 1
best_K_cross_validated_precision = sorted_cross_validated_k_precision[len(sorted_cross_validated_k_precision)-1] + 1
best_K_cross_validated_recall = sorted_cross_validated_k_recall[len(sorted_cross_validated_k_recall)-1] + 1
best_K_cross_validated_f1 = sorted_cross_validated_k_f1[len(sorted_cross_validated_k_f1)-1] + 1

print('\n')
print('Best K cross validated by accuracy is ', best_K_cross_validated_accuracy)
print('The best score achieved by best K by accuracy is ', score_list_accuracy[best_K_cross_validated_accuracy])

print('\n')
print('Best K cross validated by precision is ', best_K_cross_validated_precision)
print('The best score achieved by best K by precision is ', score_list_precision[best_K_cross_validated_precision])

print('\n')
print('Best K cross validated by recall is ', best_K_cross_validated_recall)
print('The best score achieved by best K by recall is ', score_list_recall[best_K_cross_validated_recall])

print('\n')
print('Best K cross validated by f1 is ', best_K_cross_validated_f1)
print('The best score achieved by best K by f1 is ', score_list_f1[best_K_cross_validated_f1])


# 预测
bestK = max(best_K_cross_validated_accuracy, best_K_cross_validated_precision, best_K_cross_validated_recall, best_K_cross_validated_f1)
clf = KNeighborsClassifier(n_neighbors = bestK)
clf.fit(data_now, trainLabel)
result = clf.predict(testData_now)

print(result)
df = pd.DataFrame({'PassengerId': testSet['PassengerId'], 'Survived': result})
df.to_csv('submission.csv', header = True)
