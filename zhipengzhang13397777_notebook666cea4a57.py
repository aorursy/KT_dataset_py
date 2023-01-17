import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#导入matplotlib库
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
np.random.seed(123)

%matplotlib inline
data_raw = pd.read_csv('../input/titanic/train.csv')
data_test  = pd.read_csv('../input/titanic/test.csv')

print (data_raw.info()) 
print("-"*10)
print (data_test.info()) 
data_raw.head()
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

def preprocessing(dfdata):

    dfresult= pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    sex = {'male':1,'female':0}
    dfSex = dfdata['Sex'].replace(sex)
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfdata['Age'].fillna(dfdata['Age'].median(), inplace = True)
    dfdata['AgeBin'] = pd.cut(dfdata['Age'].astype(int), 5)
    dfdata['AgeBin'] = label.fit_transform(dfdata['AgeBin'])
    dfresult = pd.concat([dfresult,dfdata['AgeBin']],axis = 1)

    #SibSp,Parch
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    
    #Fare
    dfdata['Fare'].fillna(dfdata['Fare'].median(), inplace = True)
    dfdata['FareBin'] = pd.qcut(dfdata['Fare'], 4)
    dfdata['FareBin'] = label.fit_transform(dfdata['FareBin'])
    dfresult = pd.concat([dfresult,dfdata['FareBin']],axis = 1)

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)
    
    #IsAlone
    dfdata['IsAlone'] = 1
    dfdata['IsAlone'].loc[(dfdata['SibSp'] + dfdata['Parch']) > 0] = 0
    dfresult = pd.concat([dfresult,dfdata['IsAlone']],axis = 1)
    
    #Title
    title = ['Mr', 'Miss', 'Mrs', 'Master']
    dfdata['Title'] = dfdata['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dfdata['Title'] = dfdata['Title'].apply(lambda x: x if x in title else 'Misc')
    dfdata['Title'] = label.fit_transform(dfdata['Title'])
    dftitle = pd.get_dummies(dfdata['Title'] )
    dftitle.columns = ['Title_' + str(x) for x in dftitle.columns]
    dfresult = pd.concat([dfresult,dftitle],axis = 1)
    
    return(dfresult)
x_train_total = preprocessing(data_raw)
y_train_total = data_raw[['Survived']].replace(0,-1)

x_val = preprocessing(data_test)

x_train_total.head()
from sklearn import model_selection

train_x, test_x, train_y, test_y = model_selection.train_test_split(x_train_total, y_train_total, random_state = 0)


print("train_x Shape: {}".format(train_x.shape))
print("test_x Shape: {}".format(test_x.shape))
print("train_y Shape: {}".format(train_y.shape))
print("test_y Shape: {}".format(test_y.shape))
class perceptron(object):
    def __init__(self):
        self.n_features = 0         #样本特征
        self.w = 0                  #参数w
        self.b = 0                  #参数b
        self.epoch = 0              #实际迭代次数
        self.errList = []           #错误率
    
    #求函数值
    def calY(self, X):
        return np.dot(X, self.w) + self.b
    
    def preY(self, X):
        return np.sign(self.calY(X))
    
    #训练
    def train(self, X, y, learning_rate, epochs=200):
        #初始化参数
        n_samples, self.n_features = X.shape
        self.w = np.ones((self.n_features, 1))
        self.b = 1
        for i in range(epochs):
            error = 0
            for n in range(n_samples):
                Xi = X.iloc[n]
                yi = y.iloc[n].values
                yValue = self.calY(Xi)
                #对每个样本进行判断是否分类正确
                if yi*yValue[0] <= 0:
                    self.w = self.w + learning_rate*np.expand_dims(yi,1)*np.expand_dims(Xi,1)
                    self.b = self.b + learning_rate*yi
                    error += 1
                else:
                    pass
            self.errList.append(error)
            #无错分就退出循环
            if error == 0:
                break
        self.epoch = i
        return
    
per = perceptron()
per.train(train_x, train_y, 0.0001)
print("参数是：", per.w, per.b)
print("迭代次数：", per.epoch)
#画图1：错误个数
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(per.epoch+1), per.errList)
plt.show()
from sklearn.metrics import classification_report

print(classification_report(per.preY(test_x), test_y))
repo = classification_report(per.preY(test_x), test_y, output_dict=True)
acc = repo['accuracy']
learning_rate_candidate=[0.001, 0.0001, 0.0002, 0.0005, 0.00001, 0.00002, 0.00005]

bestper = per
bestacc = acc
bestlr = 0.0001
for lr in learning_rate_candidate:
    print("testing learning rate ",lr)
    tmp_per = perceptron()
    tmp_per.train(train_x, train_y, lr)
    repo = classification_report(tmp_per.preY(test_x), test_y,output_dict=True)
    if repo['accuracy']>bestacc:
        bestacc = repo['accuracy']
        bestper = tmp_per
        bestlr = lr
print("bestlr:",lr)
print("bestacc:",acc)
y_val = pd.DataFrame(bestper.preY(x_val)).replace(-1,0).astype(int)
submit = pd.concat([data_test['PassengerId'],y_val],axis = 1)
submit.columns = ['PassengerId','Survived']
submit.to_csv("submit.csv", index=False)
submit.sample(10)


