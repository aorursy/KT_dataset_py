import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库
dataset=pd.read_csv('../input/train.csv')
testset=pd.read_csv('../input/test.csv')
dataset.columns
dataset.shape, testset.shape
dataset.head(10)
dataset.dtypes
dataset.describe()
Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()
print(Survived_m)
print(Survived_f)

df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df
#?plt.bar
df.plot(kind='bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("survived") 
plt.ylabel("count")
plt.show()
dataset['Age'].hist() 
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution')
plt.show() 

dataset[dataset.Survived==0]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of people who did not survive')
plt.show()

dataset[dataset.Survived==1]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of people who survived')
plt.show()
Survived_Age = dataset.groupby(['Survived', 'Age']).count()
Survived_Age.add_suffix('_Count').reset_index()










# 预测

# 检测模型precision， recall 等各项指标

# cross validation 找到最好的k值


# 预测


