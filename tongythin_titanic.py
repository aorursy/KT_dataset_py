import pandas as pd

import numpy as np
# 将特征变量和分类标签数值化
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection


def load_csv_data(file):
    csv_data = pd.read_csv(file)
    return csv_data


def process_data(data):
    # 补充缺失值
    simple_imputer_median = SimpleImputer(strategy="median")
    simple_imputer_constant = SimpleImputer(strategy="most_frequent")
    features = simple_imputer_median.fit_transform(data[['Pclass', 'SibSp', 'Parch']])
    embarked = simple_imputer_constant.fit_transform(data[['Embarked']])
    sex = simple_imputer_constant.fit_transform(data[['Sex']])
    age = simple_imputer_constant.fit_transform(data[['Age']])
    fare = simple_imputer_constant.fit_transform(data[['Fare']])

    # cabin = data[['Cabin']].apply(lambda x: 'YES' if str(x.values[0]) != 'nan' else 'NO', axis=1).values

    # one hot encode
    sex = pd.get_dummies(pd.DataFrame(sex)).values
    embarked = pd.get_dummies(pd.DataFrame(embarked)).values
    # cabin = pd.get_dummies(pd.DataFrame(cabin)).values

    # 分类特征转换
    ol = preprocessing.OrdinalEncoder()
    sex = ol.fit_transform(sex)

    # 标准化
    ss = preprocessing.StandardScaler()
    age = ss.fit_transform(age)
    fare = ss.fit_transform(fare)

    # sex = sex.reshape((len(sex), 1))
    features = np.hstack((features, sex, embarked, age, fare))
    print('train: ', features.shape)
    return features


def predict_survival(train_model):
    test_data = load_csv_data('/kaggle/input/titanic/test.csv')
    test_features = process_data(test_data)
    predicted = train_model.predict(test_features)
    predicted_result = list(zip(test_data['PassengerId'].values, predicted))
    df = pd.DataFrame(predicted_result, columns=['PassengerId', 'Survived'])
    # print(df.values)
    df.to_csv('gender_submission.csv', index=None)


if __name__ == '__main__':
    train_data = load_csv_data('/kaggle/input/titanic/train.csv')
    train_features = process_data(train_data)
    train_label = preprocessing.LabelEncoder().fit_transform(train_data['Survived'])

    # KNN
    knn_model = KNeighborsClassifier()
    knn_model.fit(train_features, train_label)

    # LR
    lr_model = LogisticRegression()
    lr_model.fit(train_features, train_label)

    knn_scores = model_selection.cross_val_score(knn_model, train_features, train_label, cv=10)
    knn_accuracy = np.mean(knn_scores)
    print('knn cross_val_score:', knn_scores)
    print('knn mean cross_val_score:', knn_accuracy)

    lr_scores = model_selection.cross_val_score(lr_model, train_features, train_label, cv=10)
    lr_accuracy = np.mean(lr_scores)
    print('cross_val_score:', lr_scores)
    print('mean cross_val_score:', lr_accuracy)
    predict_survival(knn_model)

