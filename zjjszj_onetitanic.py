import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Imputer
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
print(os.listdir("../input"))
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
X_train=train_data.drop(columns=['Name','Ticket','PassengerId','Survived'],axis=1)
y_train=train_data['Survived']
X_test=test_data.drop(columns=['Name','Ticket'],axis=1)

#增加属性选择器
class AddattributesSelector(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        add_SibSpParchSum = X['SibSp'] + X['Parch']
        return  np.c_[X,add_SibSpParchSum]

# #获取添加的属性与标签的相关系数
# train_data['add_SibSpParchSum']=train_data['SibSp']+train_data['Parch']
# corr_matrix = train_data.corr()
# print(corr_matrix["Survived"].sort_values(ascending=False))

#处理数据
#1 构建预处理流水线
#1.1 数值属性转换流水线
num_attribs=['Age','Fare','SibSp','Parch']
cat_attribs=['Sex','Pclass','Embarked']
num_pipeline=Pipeline([
    ('add attrib',AddattributesSelector()),
    ('num_imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler())
])
cat_pipeline=Pipeline([
    ('cat_imputer',SimpleImputer(strategy="most_frequent")),
    ('onehot',OneHotEncoder(sparse=False))
])
full_pipeline=ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',cat_pipeline,cat_attribs)
])
X_train_prepared=full_pipeline.fit_transform(X_train)
#print(X_train_prepared.shape)  （891，13）

#选择训练模型

#选择模型并微调
forest_clf=RandomForestClassifier(random_state=42)

#评估模型
#1 画ROC曲线
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
y_probas_forest=cross_val_predict(forest_clf,X_train_prepared,y_train,cv=3,method='predict_proba')
fpr, tpr, thresholds = roc_curve(y_train,y_probas_forest[:,1])
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr,"Random Forest")
plt.legend(loc="lower right", fontsize=16)
#plt.show()
s=roc_auc_score(y_train,y_probas_forest[:,1])
print(s)    #0.8281

#2 使用cross_val_score函数
forest_scores=cross_val_score(forest_clf,X_train_prepared,y_train,cv=10)
print(forest_scores.mean())     #0.8115

#测试
X_test_prepared=full_pipeline.transform(X_test)
forest_clf.fit(X_train_prepared,y_train)
y_test_predicted=forest_clf.predict(X_test_prepared)

#保存测试结果
submission = pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':y_test_predicted})
filename = 'Titanic Predictions 1.csv'
submission.to_csv(filename,index=False)

