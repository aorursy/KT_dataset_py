# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import matplotlib.cm as cm
import pandas_profiling as pp

import sklearn 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder               # 编码转换
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier          # 随机森林
from sklearn.svm import SVC, LinearSVC                       # 支持向量机
from sklearn.linear_model import LogisticRegression          # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier           # KNN算法
from sklearn.naive_bayes import GaussianNB                   # 朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier              # 决策树分类器
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier     

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('matplotlib inline')
import pandas as pd
telcom = pd.read_csv('../input/telcom/Telco-Customer-Churn.csv')
telcom = pd.read_csv('../input/telcom/Telco-Customer-Churn.csv')
telcom.head()
telcom.describe()
telcom.info()
# check null
pd.isnull(telcom).sum()
# telcom['Churn'].value_counts(normalize = True)
telcom['Churn'].value_counts()
telcom[telcom.Churn == 'No'].head()
telcom['Churn'] = telcom['Churn'].map(lambda x : 0 if x == 'No' else 1)
telcom['Churn'].value_counts()
telcom.info()
# change the dim type
telcom_raw = telcom.copy()
telcom['TotalCharges'] = pd.to_numeric(telcom.TotalCharges,errors='coerce')
telcom.TotalCharges.dtype
pd.isnull(telcom.TotalCharges).sum()
# check TotalCharges null data
telcom_raw[pd.isnull(telcom.TotalCharges)]
telcom.dropna(inplace=True)
telcom.shape
report = pp.ProfileReport(telcom)
# check churn rate
churnvalue=telcom["Churn"].value_counts()
labels=telcom["Churn"].value_counts().index

plt.pie(churnvalue,labels=labels,colors=["whitesmoke","yellow"], explode=(0.1,0),autopct='%1.1f%%', shadow=True)
plt.title("Proportions of Customer Churn")
plt.show()
# 性别、老年人、配偶、亲属对流客户流失率的影响
f, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

plt.subplot(2,2,1)
gender=sns.countplot(x="gender",hue="Churn",data=telcom,palette="Pastel2") # palette参数表示设置颜色，这里设置为主题色Pastel2
plt.xlabel("gender")
plt.title("Churn by Gender")

plt.subplot(2,2,2)
seniorcitizen=sns.countplot(x="SeniorCitizen",hue="Churn",data=telcom,palette="Pastel2")
plt.xlabel("senior citizen")
plt.title("Churn by Senior Citizen")

plt.subplot(2,2,3)
partner=sns.countplot(x="Partner",hue="Churn",data=telcom,palette="Pastel2")
plt.xlabel("partner")
plt.title("Churn by Partner")

plt.subplot(2,2,4)
dependents=sns.countplot(x="Dependents",hue="Churn",data=telcom,palette="Pastel2")
plt.xlabel("dependents")
plt.title("Churn by Dependents")
telcom.columns
# 提取特征
charges = telcom.iloc[:,1:20]
charges_columns=list(charges.columns)
print(charges_columns)
# 对特征进行编码
corrDf = charges.apply(lambda x: pd.factorize(x)[0])
corrDf_index = {}
for i in charges_columns:
    corrDf_index[i] = pd.factorize(charges[i])[1]
corrDf_index
def columns_type(df):
    '''
    获取dataset的列属性列表
    ：param df: pandas数据框
    ：return：dict(key：列类型 value：列名)
    '''
    
    type_info = {}
    for key,value in df.dtypes.items():
        value= str(value)
        if value in type_info.keys():
            type_info[value].append(key)
        else:
            type_info[value] = []
            type_info[value].append(key)
    return(type_info)
a = columns_type(telcom)
telcom[a['object']].head()
corr = corrDf.corr()
corr
plt.figure(figsize=(20,16))
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 
                 linewidths=0.2, cmap="YlGnBu",annot=True)
plt.title("Correlation between variables")
# 使用one-hot编码
tel_dummies = pd.get_dummies(telcom.iloc[:,1:21])
tel_dummies.head()
plt.figure(figsize=(15,8))
tel_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
plt.title("Correlations between Churn and variables")
# 网络安全服务、在线备份业务、设备保护业务、技术支持服务、网络电视、网络电影和无互联网服务对客户流失率的影响
covariables=["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(16,10))
for i, item in enumerate(covariables):
    plt.subplot(2,3,(i+1))
    ax=sns.countplot(x=item,hue="Churn",data=telcom,palette="Pastel2",order=["Yes","No","No internet service"])
    plt.xlabel(str(item))
    plt.title("Churn by "+ str(item))
    i=i+1
plt.show()
# 签订合同方式对客户流失率的影响
sns.barplot(x="Contract",y="Churn", data=telcom, palette="Pastel1", order= ['Month-to-month', 'One year', 'Two year'])
plt.title("Churn by Contract type")
# 付款方式对客户流失率的影响
plt.figure(figsize=(10,5))
sns.barplot(x="PaymentMethod",y="Churn", data=telcom, palette="Pastel1", order= ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check','Mailed check'])
plt.title("Churn by PaymentMethod type")
# delete customerID、PhoneService、gender
telcomvar=telcom.iloc[:,2:20]
telcomvar.drop("PhoneService",axis=1, inplace=True)

telcomvar.columns
# 提取ID
telcom_id = telcom['customerID']

telcomvar.head()
# 对客户的职位、月费用和总费用进行去均值和方差缩放，对数据进行标准化
scaler = StandardScaler(copy=False)
scaler.fit_transform(telcomvar[['tenure','MonthlyCharges','TotalCharges']])
telcomvar[['tenure','MonthlyCharges','TotalCharges']].info()
telcomvar[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(telcomvar[['tenure','MonthlyCharges','TotalCharges']])
# 使用箱线图查看数据是否存在异常值
plt.figure(figsize = (8,4))
numbox = sns.boxplot(data=telcomvar[['tenure','MonthlyCharges','TotalCharges']], palette="Set2")
plt.title("Check outliers of standardized tenure, MonthlyCharges and TotalCharges")
def uni(columnlabel):
    print(columnlabel,"--" ,telcomvar[columnlabel].unique())  # unique函数去除其中重复的元素，返回唯一值

telcomobject=telcomvar.select_dtypes(['object'])
for i in range(0,len(telcomobject.columns)):
    uni(telcomobject.columns[i])             
# 综合之前的结果来看，在六个变量中存在No internet service，即无互联网服务对客户流失率影响很小，这些客户不使用任何互联网产品，因此可以将No internet service 和 No 是一样的效果，可以使用 No 替代 No internet service
telcomvar.replace(to_replace='No internet service', value='No' ,inplace=True)
telcomvar.replace(to_replace='No phone service', value='No', inplace=True)
# 使用Scikit-learn标签编码,将分类数据转换为整数编码
def labelencode(columnlabel):
    telcomvar[columnlabel] = LabelEncoder().fit_transform(telcomvar[columnlabel])

for i in range(0,len(telcomobject.columns)):
    labelencode(telcomobject.columns[i])

for i in range(0,len(telcomobject.columns)):
    uni(telcomobject.columns[i])
telcomvar.head()
X=telcomvar
y=telcom["Churn"].values
sss=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print(sss)
print("训练数据和测试数据被分成的组数：",sss.get_n_splits(X,y))
# 建立训练数据和测试数据
for train_index, test_index in sss.split(X, y):
    print("train:", train_index, "test:", test_index)
    X_train,X_test=X.iloc[train_index], X.iloc[test_index]
    y_train,y_test=y[train_index], y[test_index]
print('原始数据特征：', X.shape,
      '训练数据特征：',X_train.shape,
      '测试数据特征：',X_test.shape)
print('原始数据标签：', y.shape,
      '   训练数据标签：',y_train.shape,
      '   测试数据标签：',y_test.shape)
Classifiers=[["Random Forest",RandomForestClassifier()],
             ["Support Vector Machine",SVC()],
             ["LogisticRegression",LogisticRegression()],
             ["KNN",KNeighborsClassifier(n_neighbors=5)],
             ["Naive Bayes",GaussianNB()],
             ["Decision Tree",DecisionTreeClassifier()],
             ["AdaBoostClassifier", AdaBoostClassifier()],
             ["GradientBoostingClassifier", GradientBoostingClassifier()],
             ["XGB", XGBClassifier()],
             ["CatBoost", CatBoostClassifier(logging_level='Silent')]  
]
Classify_result=[]
names=[]
prediction=[]
for name,classifier in Classifiers:
    classifier=classifier
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    recall=recall_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    f1 =f1_score(y_test,y_pred)
    class_eva=pd.DataFrame([recall,precision,f1])
    Classify_result.append(class_eva)
    name=pd.Series(name)
    names.append(name)
    y_pred=pd.Series(y_pred)
    prediction.append(y_pred)
names=pd.DataFrame(names)
names=names[0].tolist()
names
names=pd.DataFrame(names)
names=names[0].tolist()
result=pd.concat(Classify_result,axis=1)
result.columns=names
result.index=["recall","precision","F1"]
result
pred_X = telcomvar.tail(100)
pre_id = telcom_id.tail(100)
# 使用朴素贝叶斯方法，对预测数据集中的生存情况进行预测
model = GaussianNB()
model.fit(X_train,y_train)
pred_y = model.predict(pred_X)
pred_y_pro = model.predict_proba(pred_X)
predDf = pd.DataFrame({'customerID':pre_id, 'Churn':pred_y})
predDf
pred_y_pro = np.array(pd.DataFrame(pred_y_pro)[1])
y = telcom["Churn"].tail(100).values
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred_y_pro, pos_label = 1)
auc = sklearn.metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

