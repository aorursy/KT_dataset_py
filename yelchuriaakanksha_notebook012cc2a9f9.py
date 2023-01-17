import pandas as pd
df=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/train_data.csv')
test_data=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/test_data.csv')
sample_submission=pd.read_csv('../input/mobile-price-range-prediction-is2020-v2/sample_submission.csv')
#how does ram effected by price
import seaborn as sns
sns.jointplot(x='ram',y='price_range',data=df,color='red',kind='kde');

# Internal Memory vs PriceRange
sns.pointplot(y="int_memory", x="price_range", data=df)

# % of phones with support 3G
import matplotlib.pyplot as plt
labels = ["3G-supported",'Not supported']
values=df['three_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

# % of phones with support 4G
labels = ["4G-supported",'Not supported']
values=df['four_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

# Battery power vs Price_range
sns.boxplot(x="price_range", y="battery_power", data=df)

# No of Phones vs Camera megapixels of front and primary camera
plt.figure(figsize=(10,6))
df['fc'].hist(alpha=0.5,color='blue',label='Front camera')
df['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')

# MobileWeight vs PriceRange
sns.jointplot(x='mobile_wt',y='price_range',data=df,kind='kde');

# Talktime vs PriceRange
sns.pointplot(y="talk_time", x="price_range", data=df)

#Algorithms
x_train=df.drop(columns=['price_range','id'])
y_train=df['price_range']
x_test=test_data.drop(columns=['id'])
print(x_train,y_train,x_test,sep="\n")
from sklearn.preprocessing import StandardScaler as ss
x_trainscale=ss().fit_transform(x_train)
x_testscale=ss().fit_transform(x_test)


from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm
from sklearn.naive_bayes import GaussianNB as nvb
from sklearn.model_selection import cross_val_score as cvs

r=rfc().fit(x_trainscale,y_train)
y_pred=r.predict(x_testscale)
v=cvs(rfc(),x_trainscale,y_train,cv=3)
v

res=pd.DataFrame({'id':test_data['id'],'price_range':y_pred})
res.to_csv('/kaggle/working/res_rfc.csv',index=False)

reg=lr(C=1000,penalty='l2')
reg.fit(x_trainscale,y_train)
y_pred=reg.predict(x_testscale)
value=cvs(lr(),x_trainscale,y_train,cv=3)
print(value)

res2=pd.DataFrame({'id':test_data['id'],'price_range':y_pred})
res2.to_csv('/kaggle/working/res_lr.csv',index=False)

#decision tree algorithm


r=dtc().fit(x_trainscale,y_train)
y_pred=r.predict(x_testscale)
v=cvs(dtc(),x_trainscale,y_train,cv=3)
v


res3=pd.DataFrame({'id':test_data['id'],'price_range':y_pred})
res3.to_csv('/kaggle/working/res_dtc.csv',index=False)

#KNeighborsClassifier algorithm


r=rfc().fit(x_trainscale,y_train)
y_pred=r.predict(x_testscale)
v=cvs(rfc(),x_trainscale,y_train,cv=3)
v

res4=pd.DataFrame({'id':test_data['id'],'price_range':y_pred})
res4.to_csv('/kaggle/working/res_knn.csv',index=False)

#support vector machine algorithm


r=rfc().fit(x_trainscale,y_train)
y_pred=r.predict(x_testscale)
v=cvs(rfc(),x_trainscale,y_train,cv=3)
v

res5=pd.DataFrame({'id':test_data['id'],'price_range':y_pred})
res5.to_csv('/kaggle/working/res_svm.csv',index=False)

#naive bayes algorithm


r=rfc().fit(x_trainscale,y_train)
y_pred=r.predict(x_testscale)
v=cvs(rfc(),x_trainscale,y_train,cv=3)
v

res6=pd.DataFrame({'id':test_data['id'],'price_range':y_pred})
res6.to_csv('/kaggle/working/res_nvb.csv',index=False)
