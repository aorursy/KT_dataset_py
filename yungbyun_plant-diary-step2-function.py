import pandas as pd # 데이터 처리 모듈
import matplotlib.pyplot as plt # 데이터 시각화 모듈 
import seaborn as sns # 데이터 시각화 모듈
from sklearn.model_selection import train_test_split # 데이터 분할 모듈

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def plot(df, x_col, y_col, color_field):
    cl = df[color_field].unique()
    col = ['orange', 'blue', 'red', 'yellow', 'black', 'brown']

    fig = df[df[color_field] == cl[0]].plot(kind='scatter', x=x_col, y=y_col, color=col[0], label=cl[0])
    
    for i in range(len(cl)-1):
        df[df[color_field] == cl[i+1]].plot(kind='scatter', x=x_col, y=y_col, color=col[i+1], label=cl[i+1], ax=fig)

    fig.set_xlabel(x_col)
    fig.set_ylabel(y_col)
    fig.set_title(x_col + " vs. " + y_col)
    fig=plt.gcf()
    fig.set_size_inches(12, 7)
    plt.show()
    
def violinplot(df, a, b):
    plt.figure(figsize=(5,4))
    plt.subplot(1,1,1)
    sns.violinplot(x=a,y=b,data=df)
    
def split(df, train_s = 0.8):
    a, b = train_test_split(df, train_size = train_s)
    return a, b 

def show_files():
    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
show_files()
# CSV 파일 읽어오기
data_f = pd.read_csv('../input/plant-diary-new/plant_diary_new.csv')
# 읽어온 데이터 표시하기
plot(data_f, 'day', 'leaf_width', 'owner')
violinplot(data_f, 'owner', 'leaf_length')
# 학습용, 테스트용으로 데이터 나누기
train, test = split(data_f, 0.8)

train_X = train[['day']] # 학습 입력
train_y = train.height # 학습 정답

test_X = test[['day']] # 테스트 입력
test_y = test.height # 테스트 정답
# 머신러닝 알고리즘으로 학습 후 테스트해보기
gildong = DecisionTreeRegressor(random_state = 0)
gildong.fit(train_X, train_y)

score = gildong.score(test_X, test_y)
print('Score:', format(score,'.3f'))
print(test_X)
print(test_y)
# 알아맞춰보기
predicted = gildong.predict(test_X)
print('Predicted:', predicted)
print('Correct answer:\n', test_y)
predicted = gildong.predict([[2], [11], [60]])
print(predicted)
# 기타 다른 머신러닝 알고리즘들
gildong = KNeighborsRegressor(n_neighbors=2)


gildong = LinearRegression()


gildong = RandomForestRegressor(n_estimators=28,random_state=0)
gildong.fit(train_X, train_y)

score = gildong.score(test_X, test_y)
print('Score:', format(score,'.3f'))