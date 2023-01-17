import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data=pd.read_csv('../input/HR_comma_sep.csv')
data.left = data.left.astype(int)
data
pd.crosstab(data.salary,data.left).plot(kind='bar')
plt.show()
q=pd.crosstab(data.salary,data.left)
print(q)
print(q.sum(1))
q.div(q.sum(1),axis=0).plot(kind='bar',stacked='True')
plt.show()
观察员工满意度的分布图(histogram)
data[data.left==0].satisfaction_level.hist()
plt.show()

data[data.left==1].satisfaction_level.hist()
plt.show()
model=LogisticRegression()
y,X=dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)',data,return_type='dataframe')
X = X.rename(columns = {
    'C(sales)[T.RandD]': 'Department: Random',
    'C(sales)[T.accounting]': 'Department: Accounting',
    'C(sales)[T.hr]': 'Department: HR',
    'C(sales)[T.management]': 'Department: Management',
    'C(sales)[T.marketing]': 'Department: Marketing',
    'C(sales)[T.product_mng]': 'Department: Product_Management',
    'C(sales)[T.sales]': 'Department: Sales',
    'C(sales)[T.support]': 'Department: Support',
    'C(sales)[T.technical]': 'Department: Technical',
    'C(salary)[T.low]': 'Salary: Low',
    'C(salary)[T.medium]': 'Salary: Medium'}) 
y = np.ravel(y) # 将y变成np的一维数组
model.fit(X,y)
pd.DataFrame(list(zip(X.columns,np.transpose(model.coef_))))
print(model.score(X,y))
model.coef_
model.predict_proba([[1,0,0,1,0,0,0,0,0,0,0,0, 0.5, 0.7, 4.0, 160, 3.0, 0, 0]])
model.predict_proba(X)
pred = model.predict(X)
(abs(pred-y)).sum() / len(y)
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=0)
model2 = LogisticRegression(C=10000)
model2.fit(Xtrain, ytrain)
pred = model2.predict(Xtest)
metrics.accuracy_score(ytest, pred)

metrics.confusion_matrix(ytest, pred)
print(metrics.classification_report(ytest, pred))

print(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10))