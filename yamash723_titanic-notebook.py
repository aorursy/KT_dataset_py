import pandas as pd

import numpy as np



train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



display(train.head())

display(test.head())
# 欠損値チェック

def get_null_data_table(df):

  null_count = df.isnull().sum()

  percent = 100 * df.isnull().sum() / len(df)

  table = pd.concat([null_count, percent], axis = 1)

  return table.rename(columns = {0 : '欠損数', 1 : '%'})



display(get_null_data_table(train))

display(get_null_data_table(test))
train_c = train.copy()

test_c = test.copy()



fulldata = [train_c, test_c]



for dt in fulldata:

  # 欠損値補正

  dt["Age"] = dt["Age"].fillna(dt["Age"].median())

  dt["Fare"] = dt["Fare"].fillna(dt["Fare"].median())

  dt["Embarked"] = dt["Embarked"].fillna("S")



  # 要素追加

  dt["TotalFamirySize"] = dt["SibSp"] + dt["Parch"] + 1

  dt['IsAlone'] = 0

  dt.loc[dt['TotalFamirySize'] == 1, 'IsAlone'] = 1



  # 敬称

  dt['Salutation'] = dt.Name.str.extract(' ([A-Za-z]+).', expand=False)

  dt['Salutation'] = dt['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

  dt['Salutation'] = dt['Salutation'].replace('Mlle', 'Miss')

  dt['Salutation'] = dt['Salutation'].replace('Ms', 'Miss')

  dt['Salutation'] = dt['Salutation'].replace('Mme', 'Mrs')

 

  Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

  dt['Salutation'] = dt['Salutation'].map(Salutation_mapping) 

  dt['Salutation'] = dt['Salutation'].fillna(0)



  # カテゴリカルデータ

  dt.loc[dt["Sex"] == "male", "Sex"] = 0

  dt.loc[dt["Sex"] == "female", "Sex"] = 1

  dt.loc[dt["Embarked"] == "S", "Embarked"] = 0

  dt.loc[dt["Embarked"] == "C", "Embarked"] = 1

  dt.loc[dt["Embarked"] == "Q", "Embarked"] = 2



  display(get_null_data_table(dt))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



parameters = {

        'n_estimators'      : [10,25,50,75,100],

        'random_state'      : [0],

        'n_jobs'            : [4],

        'min_samples_split' : [5,10, 15, 20,25, 30],

        'max_depth'         : [5, 10, 15,20,25,30]

}



# モデル作成

train_fix = fulldata[0].copy()



features = train_fix[["Pclass", "Age", "Sex", "Fare", "Salutation", "TotalFamirySize", "IsAlone",  "Embarked"]].values

target = train_fix["Survived"].values



random_forest = RandomForestClassifier(

            bootstrap=True,

            class_weight=None,

            criterion='gini',

            max_depth=5,

            max_features='auto',

            max_leaf_nodes=None,

            min_samples_leaf=1,

            min_samples_split=15,

            min_weight_fraction_leaf=0.0,

            n_estimators=51,

            n_jobs=4,

            oob_score=False,

            random_state=0,

            verbose=0,

            warm_start=False)



clf = GridSearchCV(random_forest, parameters)

forest = clf.fit(features, target)





# テストデータでモデル適応

test_fix = fulldata[1].copy()



test_features = test_fix[["Pclass", "Age", "Sex", "Fare", "Salutation", "TotalFamirySize", "IsAlone",  "Embarked"]].values

my_prediction = forest.predict(test_features)



my_prediction.shape
# 提出データ作成

PassengerId = np.array(test_fix["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

my_solution.to_csv("boost.csv", index_label = ["PassengerId"])