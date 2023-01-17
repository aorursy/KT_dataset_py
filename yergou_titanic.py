
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.info()
#train.describe()
train.sample(5)
#train.isna().sum()
#train.isnull().sum()
new_train=train.drop(['Name','SibSp','Parch','Ticket','Cabin'],axis=1)

dataSample = new_train.head(20)

#dataSample[dataSample['Age'].isin(['NaN'])]         是一个DataFrame，筛选了Age中的空值
#dataSample[dataSample['Age'].isin(['NaN'])].index   上述DF的index值
#dataSample['Age'].isin(['NaN'])                     是true、false值，未筛选
#dataSample['Age'].isin(['NaN']).index               是步距为1的连续值

#dataSample[index=0]  错误
#dataSample['Age','Survived']  错误
#dataSample[['Age','Survived']] 正确
#pd.DataFrame(dataSample,['Age','Survived'])          默认为行
#pd.DataFrame(dataSample,columns=['Age','Survived'])
#pd.DataFrame(dataSample,dataSample['Age'].dropna().index)   默认为行
df=pd.DataFrame(dataSample,index=dataSample['Age'].dropna().index)
df['Sex']
#df.Sex
dataSample.loc[:,['Age','Survived']]
#行索引
#df.loc[1,3]  错误
#df.loc['1','3'] 错误
#df.loc[['1','3']]  错误
#df.loc[[1,3]] 正确
#df.loc[['1':'3']] 错误
#df.loc['1':'3']
#df.index  结果显示dtype='int64'，loc的索引是按照label识别的
import seaborn as sns
import matplotlib.pyplot as plt
def test(df):
    dfData = df.corr()
    plt.subplots(figsize=(10, 8)) # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap=plt.cm.RdBu)
    #plt.savefig('./BluesStateRelation.png')
    plt.show()
test(train)