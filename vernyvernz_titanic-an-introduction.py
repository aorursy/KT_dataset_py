import pandas as pd
warnings.filterwarnings("ignore")
#start by import data into jupyter



path = "C:/Users/Admin/Documents/data/titanic/"

train = pd.read_csv(path+"train.csv")

test = pd.read_csv(path+"test.csv")
train
#select features which could be valueable



train_clean = train[["Pclass","Sex","Age","SibSp","Parch","Fare"]]

train_clean
#notice that the 'sex' column can be dummy coded



train_cleaner = train_clean.copy()

train_cleaner

train_cleaner.columns = ["Class","Female","Age","SibSp","ParCh","Fare"]



#dummy coding

for i in range(len(train_cleaner)):

    if train_cleaner["Female"][i] == "female":

        train_cleaner["Female"][i] = 1

    else:

        train_cleaner["Female"][i] = 0

        

train_cleaner
#lets see how much of our data is missing

len(train_cleaner) - len(train_cleaner.dropna())
177/len(train_cleaner)
# so it turns out that 20% of our training data is missing

#thats a bit too much for my liking
train_cleaner2 = train_cleaner.fillna(0)

train_cleaner2
#lets visualise our data a bit



#first some helpful numbers

class1 = train_cleaner2[train_cleaner["Class"] == 1]

c1 = len(class1)

class2 = train_cleaner[train_cleaner["Class"] == 2]

c2 = len(class2)

class3 = train_cleaner[train_cleaner["Class"] == 3]

c3 = len(class3)

total = len(train_cleaner2)
plt.pie([(c1/total)*100,(c2/total)*100,(c3/total)*100] , labels = ["1st class","2nd class","3rd class"] , explode = (0.2,0.3,0) , shadow = True , autopct = "%1.1f%%")

plt.title("Passengers boarding")

plt.legend()

plt.show()
#from the pie chart we can see how many people from each class boarded the ship

#with those numbers we can then see how many from each class survived
class1 = train[train["Pclass"] == 1]

class1_survived = len(class1[class1["Survived"] == 1])



class2 = train[train["Pclass"] == 2]

class2_survived = len(class2[class2["Survived"] == 1])



class3 = train[train["Pclass"] == 3]

class3_survived = len(class3[class3["Survived"] == 1])
plt.pie([(class1_survived/c1)*100,(class2_survived/c2)*100,(class3_survived/c3)*100] , labels = ["1st class","2nd class","3rd class"] , explode = (0.01,0.56,0.4),shadow = True , autopct = "%1.1f%%")

plt.legend()

plt.title("Survivors from each class")

plt.show()
#lets train our learner with the following training_set



train_data = train_cleaner2

train_data
train_target = train["Survived"]

train_target
#lets import the machine learning module

from sklearn import neighbors

import math
knn = neighbors.KNeighborsClassifier(n_neighbors = int(math.sqrt(len(train_data))))
knn
knn.fit(train_data,train_target)
#now that we have trained our learner , lets start making predictions

#we define a function that takes a df and outputs a prediction in a more human readable format



def k_neighbors_implementation( test_set ):

    predictions = knn.predict(test_set)

    

    temp = test_set.copy()

    temp["Survived"] = predictions

        

    return temp
test
#lets clean the data a bit b4 feeding to knn



test_ = test.drop(["PassengerId","Name","Ticket","Cabin","Embarked"] , axis = 1)

test_
for i in range(len(test_)):

    if test_["Sex"][i] == "female":

        test_["Sex"][i] = 1

    else:

        test_["Sex"][i] = 0
test_clean = test_.fillna(1)

test_clean
knn.predict(test_clean)
#voila , it works
test_clean
results = k_neighbors_implementation(test_clean)
results
print("")