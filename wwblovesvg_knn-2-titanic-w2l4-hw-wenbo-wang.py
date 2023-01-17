import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# function to read data
def load_data(data_dir):
    train = pd.read_csv(data_dir + "train.csv", header = 0, sep = ',')
    test = pd.read_csv(data_dir + "test.csv", header = 0, sep = ',')
    print(train.head())
    print(test.head())
    print(train.shape, test.shape)
    return train, test

data_dir = "../input/"
train, test = load_data(data_dir)
print(train.columns)
print(train.dtypes)
print(train.info())
print(train.describe())
print(train.head(20))
for col in train.columns:
    print("{} has any NaN? {}".format(col, train[col].isnull().values.any()))


print(train.Sex.value_counts())
survived_male = train.Survived[train.Sex == "male"].value_counts()
survived_female = train.Survived[train.Sex == 'female'].value_counts()
df = pd.DataFrame({"male": survived_male, "female": survived_female})
df = df.rename(index = {1:"Life", 0:"Death"}, inplace = False)
df.plot(kind = "bar", stacked = True)
plt.title("Survival by Sex")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()
train.Age.hist()
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

train.Age[train.Survived == 0].hist()
plt.title("Age distribution of people who did not survive")
plt.xlabel("Age distribution")
plt.ylabel("Count")
plt.show()

train.Age[train.Survived == 1].hist()
plt.title("Age distribution of people who survived")
plt.xlabel("Age distribution")
plt.ylabel("Count")
plt.show()
train.Fare.hist()
plt.title("Fare distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

train.Fare[train.Survived == 0].hist()
plt.title("Fare distribution of people who did not survive")
plt.xlabel("Fare distribution")
plt.ylabel("Count")
plt.show()

train.Fare[train.Survived == 1].hist()
plt.title("Fare distribution of people who survived")
plt.xlabel("Fare distribution")
plt.ylabel("Count")
plt.show()
train.Pclass.hist()
plt.title("Pclass distribution")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()

survived_p1 = train.Survived[train.Pclass == 1].value_counts()
survived_p2 = train.Survived[train.Pclass == 2].value_counts()
survived_p3 = train.Survived[train.Pclass == 3].value_counts()

df = pd.DataFrame({"P1": survived_p1, "P2": survived_p2, "P3": survived_p3})
df = df.rename(index = {1: "Life", 0: "Death"}, inplace = False)

df.plot(kind = "bar", stacked = True)
plt.title("Survival by Pclass")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()
survived_q = train.Survived[train.Embarked == "Q"].value_counts()
survived_s = train.Survived[train.Embarked == "S"].value_counts()
survived_c = train.Survived[train.Embarked == "C"].value_counts()

df = pd.DataFrame({"Q": survived_q, "S": survived_s, "C": survived_c})
df = df.rename(index = {1: "Life", 0: "Death"}, inplace = False)

df.plot(kind = "bar", stacked = True)
plt.title("Survival by Embarked place")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()
labels = ['Pclass', 'Sex', "Age", "Fare", "Embarked"]
temp_x_train = train[labels]
temp_y_train = train.Survived
temp_x_test = test[labels]

print(temp_x_train.shape, temp_x_test.shape)
# first see the sex and embarked counts
print(train.Sex.value_counts())
print(train.Embarked.value_counts())
# define a function to deal with NaN
def fill_nan(data):
    copy = data.copy(deep = True)
    
    # get medians
    age_median = copy["Age"].median()
    pclass_median = copy["Pclass"].median()
    fare_median = copy["Fare"].median()
    sex_median = "male"
    embarked_median = "S"
    
    # fill with median
    copy["Age"] = copy["Age"].fillna(age_median)
    copy["Pclass"] = copy["Pclass"].fillna(pclass_median)
    copy["Fare"] = copy["Fare"].fillna(fare_median)
    copy["Sex"] = copy["Sex"].fillna(sex_median)
    copy["Embarked"] = copy["Embarked"].fillna(embarked_median)
    
    return copy

fillna_x_train, fillna_x_test = fill_nan(temp_x_train), fill_nan(temp_x_test)
print(fillna_x_train.head(20))
print(fillna_x_train.isnull().values.any(), fillna_x_test.isnull().values.any())
# transfer sex to numerical values
def transfer_sex(data):
    copy = data.copy(deep = True)
    
    # turn male to 1, female to 0
    copy.loc[data.Sex == "male", "Sex"] = 1
    copy.loc[data.Sex == 'female', "Sex"] = 0
    
    return copy

sex_x_train, sex_x_test = transfer_sex(fillna_x_train), transfer_sex(fillna_x_test)
print(sex_x_train.head(10))
print(sex_x_test.head(10))
# make a function to turn embarked alphabetical values into numerical values
def transfer_embarked(data):
    copy = data.copy(deep = True)
    
    copy.loc[copy.Embarked == "Q", "Embarked"] = 1
    copy.loc[copy.Embarked == "S", "Embarked"] = 2
    copy.loc[copy.Embarked == 'C', "Embarked"] = 3
    
    return copy

embarked_x_train, embarked_x_test = transfer_embarked(sex_x_train), transfer_embarked(sex_x_test)
print(embarked_x_train.shape, embarked_x_test.shape)
print(embarked_x_train.head(10))
print(embarked_x_test.head(10))
# generate original train and test data set
origin_x_train, origin_y_train, origin_x_test = embarked_x_train, temp_y_train, embarked_x_test
print(origin_x_train.shape, origin_y_train.shape, origin_x_test.shape)
print(origin_y_train[:10])
from sklearn.model_selection import train_test_split
x_train, x_vali, y_train, y_vali = train_test_split(origin_x_train, origin_y_train, test_size = 0.2, random_state = 0)
print(x_train.shape, x_vali.shape, y_train.shape, y_vali.shape)
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
# 预测
k_range = range(1, 51)
scores = list()
for k in k_range:
    start = time.time()
    print("k = {} now starts...".format(k))
    
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_vali)
    
    accuracy = accuracy_score(y_vali, y_pred)
    scores.append(accuracy)
    print(confusion_matrix(y_vali, y_pred))
    print(classification_report(y_vali, y_pred))
    
    end = time.time()
    print("k = {} now ends, time spent = {}".format(k, end - start))
# 检测模型precision， recall 等各项指标
plt.title("Accuracy by k values")
plt.plot(k_range, scores)
plt.xlabel("K value")
plt.ylabel("Accuracy score")
plt.show()
sorted = np.array(scores).argsort()
best_accuracy = scores[sorted[-1]]
best_k = sorted[-1] + 1
print("best accuracy = {}, best k = {}".format(best_accuracy, best_k))
# 预测
knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(origin_x_train, origin_y_train)
final_y_pred = knn.predict(origin_x_test)
print(final_y_pred)

df = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": final_y_pred})
print(df.head(20))
df.to_csv("submission.csv", header = True, index = False)