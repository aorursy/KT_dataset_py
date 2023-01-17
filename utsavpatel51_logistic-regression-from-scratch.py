import numpy as np

import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score,confusion_matrix , precision_score,recall_score

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
iris = load_iris()

#make data ready for binary classification

#take only first 2 feature and the two non-linearly separable classes are labeled with the same category

X = iris.data[:,:2]

y = (iris.target != 0) * 1
plt.scatter(x=X[y==0][:,0],y=X[y==0][:,1],color='red',label='0')

plt.scatter(x=X[y==1][:,0],y=X[y==1][:,1],color='green',label='1')

plt.legend()
class LogisticRegression_o:

    def __init__(self,lr=0.01,num_itre = 5000,fit_intercept=True,verbose=False):

        self.lr = lr

        self.num_itre = num_itre

        self.fit_intercept =fit_intercept

        self.verbose = verbose

        

    def __add_intercept(self):

        intercept = np.ones((X.shape[0],1))

        return np.concatenate((intercept,X) , axis=1)

    

    def __sigmoid(self,z):

        return 1/(1+np.exp(-z))

    

    def __loss(self,hypo,y):

        return (-y*np.log(hypo) + (1-y)*np.log(1-hypo)).mean()

    

    def fit(self,X,y):

        if self.fit_intercept:

            X = self.__add_intercept()

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_itre):

            z = np.dot(X,self.theta)

            hypo = self.__sigmoid(z)

            gre = np.dot(hypo-y , X) / y.shape[0]

            self.theta = self.theta - self.lr* gre

            if self.verbose==True and i%10000 == 0:

                z = np.dot(X,self.theta)

                hypo = self.__sigmoid(z)

                print(f'loss: {self.__loss(hypo)}')

                

    def predict_prob(self,X):

        if self.fit_intercept:

            X = self.__add_intercept()

        z = np.dot(X,self.theta)

        return self.__sigmoid(z)

    

    def predict(self,X,threshold=0.5):

        return self.predict_prob(X) >= threshold
lr = LogisticRegression_o(lr=0.1,num_itre=300000)

lr.fit(X,y)

pre = lr.predict(X)

(pre == y).mean()
t = lr.theta

t
plt.scatter(x=X[y==0][:,0],y=X[y==0][:,1],color='red',label='0')

plt.scatter(x=X[y==1][:,0],y=X[y==1][:,1],color='green',label='1')

plt.legend()



#plot θ.T * X = 0

#plot the line defined by theta[0] + theta[1]*x + theta[2]*y = 0



de_x = np.linspace(4,8,50)

de_y = -(t[0]+t[1] * de_x)/t[2]

plt.plot(de_x,de_y)
#using sklearn libray

X = iris.data[:,:2]

y = (iris.target != 0) * 1

model = LogisticRegression(C=1e20,solver='liblinear')

model.fit(X,y)

pre1 = model.predict(X)

(y==pre1).mean()



plt.scatter(x=X[y==0][:,0],y=X[y==0][:,1],color='red',label='0')

plt.scatter(x=X[y==1][:,0],y=X[y==1][:,1],color='green',label='1')

plt.legend()



#plot θ.T * X = 0

#plot the line defined by theta[0] + theta[1]*x + theta[2]*y = 0



de_x = np.linspace(4,8,50)

de_y = -(model.intercept_+model.coef_[0][0] * de_x)/model.coef_[0][1]

plt.plot(de_x,de_y)
heart_data = pd.read_csv('../input/heart-disease-uci/heart.csv')

#heart_data=heart_data.sample(frac=1) #shuffle the data



X = heart_data.iloc[:,:-1]

y = heart_data.iloc[:,-1]

print(heart_data['target'].value_counts())
print(heart_data.isna().sum())
heart_data.head()
import seaborn as sns

plt.figure(figsize=(12,13))

cor = heart_data.corr().round(2)

#sns.heatmap(data=cor,annot=True)
#handle categorical data

def coder(X_1,col_name):

    label = LabelEncoder()

    X_1 = label.fit_transform(X_1)

    X_1 = X_1.reshape(len(X_1),1)

    one = OneHotEncoder(categories='auto',sparse=False)

    X_1 = one.fit_transform(X_1)

    df = pd.DataFrame(X_1)

    df = rename_col(df,col_name)

    return df

def rename_col(df,col_name):

    col_name_list=[]

    for i in range(len(df.columns)):

        col_name_list.append(col_name+'_'+ str(i))

    df.columns = col_name_list

    return df
new_sex = coder(heart_data['sex'],'sex').astype(float)

new_cp = coder(heart_data['cp'],'cp').astype(float)

new_fbs = coder(heart_data['fbs'],'fbs').astype(float)

new_exang = coder(heart_data['exang'],'exang').astype(float)

new_slope = coder(heart_data['slope'],'slope').astype(float)

new_ca = coder(heart_data['ca'],'ca').astype(float)

new_thal = coder(heart_data['thal'],'thal').astype(float)

new_restecg = coder(heart_data['restecg'],'restecg').astype(float)

new_age = heart_data['age'].astype(float)

new_trestbps = heart_data['trestbps'].astype(float)

new_chol = heart_data['chol'].astype(float)

new_oldpeak = heart_data['oldpeak'].astype(float)

new_thalach = heart_data['thalach'].astype(float)
#New X

X = pd.concat([new_age,new_sex,new_trestbps,new_chol,new_fbs,new_restecg,new_thalach,new_exang,

              new_oldpeak,new_slope,new_ca,new_thal],axis=1)

X.columns

X.head()
from sklearn.preprocessing import MinMaxScaler

pre = MinMaxScaler()

X = pre.fit_transform(X)

X = pd.DataFrame(X)

X.head()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=100,shuffle=True)

X_train.shape , X_test.shape
#select diffrent method

def cal(method,c=1,p='l2'):

    lr = LogisticRegression(C=c,solver=method,penalty=p)

    lr.fit(X_train,y_train)

    pre_test = lr.predict(X_test)

    pre_train = lr.predict(X_train)

    train_score = accuracy_score(y_train,pre_train)

    test_score = accuracy_score(y_test,pre_test)

    cm = confusion_matrix(y_test,pre_test)

    

    TP = cm[1][1]

    FP = cm[0][1]

    FN = cm[1][0]

    TN = cm[0][0]

    Precision = precision_score(y_test,pre_test)

    Recall = recall_score(y_test,pre_test)

    return train_score,test_score,TP,FP,FN,TN,Precision,Recall

method = ['newton-cg','lbfgs','liblinear','sag','saga']

train_list , test_list , TP_list,FP_list,FN_list,TN_list,Precision_list,Recall_list =[],[],[],[],[],[],[],[]
for i in method:

    method_df = pd.DataFrame()

    train_score,test_score,TP,FP,FN,TN,Precision,Recall = cal(i)

    train_list.append(train_score)

    test_list.append(test_score)

    TP_list.append(TP)

    FP_list.append(FP)

    FN_list.append(FN)

    TN_list.append(TN)

    Precision_list.append(Precision)

    Recall_list.append(Recall)

method_df['method'] = method

method_df['train_score'] = train_list

method_df['test_score'] = test_list

method_df['TP'] = TP_list

method_df['FP'] = FP_list

method_df['FN'] = FN_list

method_df['TN'] = TN_list

method_df['Precision'] = Precision_list

method_df['Recall'] = Recall_list

method_df
N=5

ind = np.arange(N)

width=0.35

plt.bar(ind,train_list,width,color='b',label='Train acc')

plt.bar(ind+width,test_score,width,color='r',label='Test acc')

plt.ylabel('acc')

plt.xticks(ind+width/2,(method))

plt.show()
#select diff C value with liblinear

train_list , test_list , TP_list,FP_list,FN_list,TN_list,Precision_list,Recall_list =[],[],[],[],[],[],[],[]

c_value=[0.0001,0.001,0.01,0.1,1,2,3,5,9,10,20,30,40,50]

for i in c_value:

    c_df = pd.DataFrame()

    train_score,test_score,TP,FP,FN,TN,Precision,Recall = cal(method='liblinear',c=i)

    train_list.append(train_score)

    test_list.append(test_score)

    TP_list.append(TP)

    FP_list.append(FP)

    FN_list.append(FN)

    TN_list.append(TN)

    Precision_list.append(Precision)

    Recall_list.append(Recall)

c_df['c_value'] = c_value

c_df['train_score'] = train_list

c_df['test_score'] = test_list

c_df['TP'] = TP_list

c_df['FP'] = FP_list

c_df['FN'] = FN_list

c_df['TN'] = TN_list

c_df['Precision'] = Precision_list

c_df['Recall'] = Recall_list

c_df
%matplotlib inline

plt.plot(c_value,train_list,'r-',label='train acc')

plt.plot(c_value,test_list,'b-',label='test acc')

plt.xlabel('C parm')

plt.ylabel('acc')

plt.legend()

plt.show()