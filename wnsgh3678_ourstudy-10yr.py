import pandas as pd # 데이터 처리를 위한 모듈

lst  = []







wowco2 = pd.read_csv('../input/data-for-predict-10/co2dtp10.csv')

wowhealth = pd.read_csv('../input/data-for-predict-10/healthdtp10.csv')

wowprice = pd.read_csv('../input/data-for-predict-10/pricedtp10.csv')



for i in range(70):

    lstj = []

    for j in range(2019,2070,10):

        lstk = []

        for k in (wowhealth,wowco2,wowprice):

            lstk.append(k[str(j)][i])

        lstj.append(lstk)

    lst.append(lstj)



#print(lst)







import pandas as pd # 데이터 처리를 위한 모듈

import seaborn as sns # 데이터 시각화 모듈

import matplotlib.pyplot as plt # 데이터 시각화 모듈 



# CSV 파일 읽어오기

a = "../input/standard-predict/chtw4.csv"

wow = pd.read_csv(a)



from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘

#from sklearn.cross_validation import train_test_split # 데이타 쪼개주는 모듈 

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm



train, test = train_test_split(wow, test_size = 0.8)# train=70% and test=30%

#print(train.shape)

#print(test.shape)



train_X = train[["health","co2","price"]]

train_y = train.grade # 정답 선택



#print(test_y)



import warnings  

warnings.filterwarnings('ignore')



#결정 트리(Decision Tree)

baby3 = DecisionTreeClassifier()

baby3.fit(train_X,train_y)



prdtlst = []

for i in range(70):

    prdtlstj = []

    for j in range(6):

        test_X = [lst[i][j]]

        prediction = baby3.predict(test_X)

        prdtlstj.extend(prediction)

    prdtlst.append(prdtlstj)



#print(prdtlst)



yrlst = []

for i in range(2019,2070,5):

    yrlst.append(i)



for i in range(70):

    print(wowco2.국가별[i])

    plt.plot(prdtlst[i])

    plt.axis([-1,6,0,5])

    plt.show()
