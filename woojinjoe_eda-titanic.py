import pandas as pd
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.head()
train_data.info()
# Ref: https://github.com/wikibook/ml-definitive-guide/blob/master/2%EC%9E%A5/2.6%20%EC%82%AC%EC%9D%B4%ED%82%B7%EB%9F%B0%EC%9C%BC%EB%A1%9C%20%EC%88%98%ED%96%89%ED%95%98%EB%8A%94%20%ED%83%80%EC%9D%B4%ED%83%80%EB%8B%89%20%EC%83%9D%EC%A1%B4%EC%9E%90%20%EC%98%88%EC%B8%A1%20.ipynb

from sklearn.preprocessing import LabelEncoder



# Null 처리 함수

def fillna(df):

    df['Age'].fillna(df['Age'].mean(),inplace=True)

    df['Cabin'].fillna('N',inplace=True)

    df['Embarked'].fillna('N',inplace=True)

    df['Fare'].fillna(0,inplace=True)

    return df



# 머신러닝 알고리즘에 불필요한 속성 제거

def drop_features(df):

    df.drop(['PassengerId', 'Name','Ticket'],axis=1,inplace=True)

    return df



# 레이블 인코딩 수행. 

def format_features(df):

    df['Cabin'] = df['Cabin'].str[:1]

    features = ['Cabin','Sex','Embarked']

    for feature in features:

        le = LabelEncoder()

        le = le.fit(df[feature])

        df[feature] = le.transform(df[feature])

    return df



# 앞에서 설정한 Data Preprocessing 함수 호출

def transform_features(df):

    df = fillna(df)

    df = drop_features(df)

    df = format_features(df)

    return df
y = train_data["Survived"]

print("y shape:", y.shape)
passengers = test_data['PassengerId']

passengers.head()
merged_data = pd.concat([train_data, test_data], sort=False)

processed_data = transform_features(merged_data.drop('Survived',axis=1))



X = processed_data.iloc[:891, :]

X.info()



test_data = processed_data.iloc[891:, :]

test_data.info()
# from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score



def eval_pred(y_test, pred):

    print('accuracy: ', accuracy_score(y_test, pred))

    print('f1: ', f1_score(y_test, pred))

    print('roc auc:', roc_auc_score(y_test, pred))





# X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3)



parameters = {'num_iterations': list(range(100, 300, 100)),

              'num_leaves':list(range(32, 48, 6)),

              'min_child_samples':list(range(20, 30, 2)),

              'max_depth':list(range(10, 20, 2))

             }

print(parameters)
import warnings

warnings.filterwarnings('ignore')



# lgbm = LGBMClassifier()

# for tuning

# grid_dclf = GridSearchCV(lgbm , param_grid=parameters , scoring='roc_auc', cv=5, verbose=10)

# grid_dclf.fit(X , y)

# print('GridSearchCV 최적 하이퍼 파라미터 :',grid_dclf.best_params_)

# print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dclf.best_score_))

# best_dclf = grid_dclf.best_estimator_
final_model = LGBMClassifier(max_depth=16, min_child_samples=26, num_iterations=100, num_leaves=32)

final_model.fit(X, y)
predictions = final_model.predict(test_data)

output = pd.DataFrame({'PassengerId': passengers, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")