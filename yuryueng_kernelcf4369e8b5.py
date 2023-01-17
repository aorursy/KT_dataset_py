# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load train set

#train set을 불러온다

df = pd.read_csv('../input/train.csv')

df
#simple descriptive statistics : histogram

#나이에 대한 간단한 histogram을 본다

import matplotlib.pyplot as plt

df['Age'].hist(bins=20)
#fast feature engineering

#빠른 feature engineering



#conpcept : remove every categorical variables, every None/nan variables, one-hot about 'Sex'

#모토는 일단 categorical variable은 전부 제거 + None/nan은 전부 제거, 'Sex'만 One-hot함



#1. Drop all categorical variables.

#1. Categorical Data라서 당장 처리가 어려운 애들은 전부 짤라낸다.

simple_feature_cutting_df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)



#2. drop every None/nans

#2. None이나 nan은 다 짤라낸다

simple_feature_cutting_df = simple_feature_cutting_df.dropna()



#3. one-hot encode feature 'Sex'

#3. 최소한 성별 정도는 one-hot encoding한다

simple_feature_cutting_df = pd.get_dummies(simple_feature_cutting_df, columns=['Sex'])



#4. show result. you can compare upper dataframe and this dataframe

#4. 학습용 결과물 출력. 위쪽 셀의 raw data와 비교해볼 수 있을 것이다.

simple_feature_cutting_df
#split data set to train/test

#데이터 셋을 test / train 으로 나눔



#test = front 100.

#test = 맨앞100개

test_data_set = simple_feature_cutting_df[:100]



#train = all others.

#train = 나머지 다

train_data_set = simple_feature_cutting_df[100:]
#using random forest classifier model

#모델로 사이킷런 랜덤포레스트를 쓸 예정

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()



#label data is 'Survived'

#정답데이터는 아까만든 train_data_set중 Survived column을 사용

label_data = train_data_set['Survived']



#train data is all but 'Survived' column

#학습데이터는 아까만든 train_data_set중  Survived만 빼버리고 나머지를 사용

train_data = train_data_set.drop('Survived', axis=1)



#let's do learn(fitting)

#학습

model.fit(train_data, label_data)
#predict test dataset with fitted model. using all test data but 'Survived' column.

#학습된 모델로 테스트 셋에서 똑같이 'Survived Column'을 뺀 값을 넣어봐서 예측을 해본다.

result_test_predict = model.predict(test_data_set.drop('Survived', axis=1))



#get real observation target value ('Survived column') ... to compare predict is how much right.

#실제 테스트 셋의 observation target value를 가져온다... 왜냐면 정답을 맞췄는지 봐야하니까.

real_test_observations = np.array(test_data_set['Survived'])



#make compare dataframe and show

#두개를 비교해볼수 있게 dataframe만들어서 얼마나 맞았는지 뿌려본다.

result = pd.DataFrame({'predict':result_test_predict, 'real':real_test_observations } )

result['Correct'] = result.apply(lambda row : row['predict'] == row['real'], axis=1)

result
#correct count

#맞춘 갯수

num_correct = len(result[result['Correct']]) 



#all count

#모든 갯수

num_total = len(result)



#accuracy

num_correct / num_total