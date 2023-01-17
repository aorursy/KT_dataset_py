import pandas as pd # 데이터 처리 모듈

import matplotlib.pyplot as plt # 데이터 시각화 모듈 

import seaborn as sns # 데이터 시각화 모듈

from sklearn.model_selection import train_test_split # 데이터 분할 모듈

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics #for checking the model accuracy



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
# CSV 파일 읽어오기

data_f = pd.read_csv("../input/personal-dataset/personal_dataset.csv")
#plot(data_f, "Height", "Weight", "Sex")

#violinplot(data_f, 'Sex', 'Height')
train, test = split(data_f, 0.8)



train_X = train[['Height','FeetSize','Weight']] # 학습 입력

train_y = train.Sex # 학습 정답



test_X = test[['Height','FeetSize','Weight']] # 테스트 입력

test_y = test.Sex # 테스트 정답
youngja = DecisionTreeClassifier()

youngja.fit(train[['Height','FeetSize','Weight']],train_y)



prediction = youngja.predict(test_X)

print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)
from sklearn import svm

from sklearn.linear_model import LogisticRegression 

from sklearn.neighbors import KNeighborsClassifier 



gildong = svm.SVC()  

gildong = LogisticRegression()

gildong = KNeighborsClassifier(n_neighbors=3)