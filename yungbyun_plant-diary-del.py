import pandas as pd # 데이터 처리 모듈

import matplotlib.pyplot as plt # 데이터 시각화 모듈 

import seaborn as sns # 데이터 시각화 모듈

from sklearn.model_selection import train_test_split # 데이터 분할 모듈



from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
a = 3

print(a)
import pandas as pd # 데이터 처리 모듈

import matplotlib.pyplot as plt # 데이터 시각화 모듈 

import seaborn as sns # 데이터 시각화 모듈

from sklearn.model_selection import train_test_split # 데이터 분할 모듈



from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



# data_f = load_csv('../input/plant-diary-new/plant_diary_new.csv')

def load_csv(file):

    import pandas as pd # 데이터 처리 모듈

    # CSV 파일 읽어오기

    imsi = pd.read_csv(file)

    return imsi;



def show_files(f):

    import os

    for dirname, _, filenames in os.walk(f):

        for filename in filenames:

            return os.path.join(dirname, filename)



# plot(data_f, 'day', 'leaf_length', 'owner')

def plot(df, _x, _y, _color_filed):

    # 읽어온 데이터 표시하기

    cl = df[_color_filed].unique()



    col = ['orange', 'blue', 'red', 'yellow', 'black', 'brown']



    fig = df[df[_color_filed] == cl[0]].plot(kind='scatter', x=_x, y=_y, color=col[0], label=cl[0])



    for i in range(len(cl)-1):

        df[df[_color_filed] == cl[i+1]].plot(kind='scatter', x=_x, y=_y, color=col[i+1], label=cl[i+1], ax=fig)



    fig.set_xlabel(_x)

    fig.set_ylabel(_y)

    fig.set_title(_x + " vs. " + _y)

    fig=plt.gcf()

    fig.set_size_inches(12, 7)

    plt.show()



# violin_plot(data_f, 'owner', 'leaf_length')

def violin_plot(df, _x, _y):

    plt.figure(figsize=(5,4))

    plt.subplot(1,1,1)

    sns.violinplot(x=_x,y=_y,data=df)

    

# heatmap(df, ['day', 'height', 'leaf_width', 'leaf_length', 'owner'])

def heatmap(dataf, cols):

    plt.figure(figsize=(12,8))

    sns.heatmap(data_f[cols].corr(),annot=True)



# show_cols(data_f)    

def show_cols(df):

    for col in df.columns: 

        print(col) 

        

# boxplot('owner', 'height')

def boxplot(a, b):

    f, sub = plt.subplots(1, 1,figsize=(7,5))

    sns.boxplot(x=data_f[a],y=data_f[b], ax=sub)

    sub.set(xlabel=a, ylabel=b);

    

# plot_3d('day', 'leaf_length', 'leaf_width')

def plot_3d(a, b, c):

    from mpl_toolkits.mplot3d import Axes3D



    fig=plt.figure(figsize=(12,8))



    ax=fig.add_subplot(1,1,1, projection="3d")

    ax.scatter(data_f[a],data_f[b],data_f[c],c="blue",alpha=.5)

    ax.set(xlabel=a,ylabel=b,zlabel=c)

    

def get_score_4_algo(a, b, c, d):

    # [1] 결정트리 예측기 머신러닝 알고리즘 학습

    gildong = DecisionTreeRegressor(random_state = 0)

    gildong.fit(train_X, train_y) #학습용 문제, 학습용 정답  

    

    # 점수 계산

    score1 = gildong.score(test_X, test_y) # 시험 문제, 시험 정답

    #print('Score:', format(score,'.3f'))

    # score의 의미: 정확하게 예측하면 1, 평균으로 예측하면 0, 더 못 예측하면 음수  



    # [2] 랜덤 포레스트 예측기 머신러닝 알고리즘

    youngja = RandomForestRegressor(n_estimators=28,random_state=0)

    youngja.fit(train_X, train_y)



    score2 = youngja.score(test_X, test_y)

    #print('Score:', format(score,'.3f'))

    

    # [3] K근접이웃 예측기 머신러닝 알고리즘

    cheolsu = KNeighborsRegressor(n_neighbors=2)

    cheolsu.fit(train_X, train_y)



    score3 = cheolsu.score(test_X, test_y)

    #print('Score:', format(score,'.3f'))

    

    # [4] 선형회귀 머신러닝 알고리즘

    minsu = LinearRegression()

    minsu.fit(train_X, train_y)



    score4 = minsu.score(test_X, test_y)

    #print('Score:', format(score,'.3f')) 

    

    plt.plot(['DT','RF','K-NN','LR'], [score1, score2, score3, score4])

    



def split_4_parts(df, munje_cols, dap_col):

    # 학습용(문제, 정답), 테스트용(문제, 정답)으로 데이터 나누기

    train, test = train_test_split(df, train_size = 0.8)



    # 학습용 문제와 정답

    a = train[[munje_cols]]

    b = train[[dap_col]]



    # 시험 문제와 정답

    c = test[[munje_cols]]

    d = test[[dap_col]]

    

    return a, b, c, d
print(show_files('/kaggle/input'))
!pwd
data_f = load_csv('../input/plant-diary-new/plant_diary_new.csv')
plot(data_f, 'day', 'leaf_length', 'owner')
violin_plot(data_f, 'owner', 'leaf_length')
print(data_f)
show_cols(data_f)
# 특징별 상관관계 분석 맵

heatmap(data_f, ['day', 'height', 'leaf_width', 'leaf_length', 'owner'])
boxplot('owner', 'height')
plot_3d('day', 'leaf_length', 'leaf_width')
data_f
train_X, train_y, test_X, test_y = split_4_parts(data_f, 'day', 'height')
print(train_X)

print('---')

print(test_X)
scores = get_score_4_algo(train_X, train_y, test_X, test_y)