import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
print(os.listdir("../input"))
df=pd.read_csv('../input/insurance.csv')
d={'yes': 1, 'no': 0}
df['smoker']=df['smoker'].map(d)
d2={'female': 1, 'male':0}
df['sex']=df['sex'].map(d2)
print(df.head())
print(df.info())
normal=np.logical_and(25.0>df['bmi'].values, df['bmi'].values>20.0)
overweight=np.logical_and(30>df['bmi'].values, df['bmi'].values>25)
obese=np.logical_and(60>df['bmi'].values, df['bmi'].values>30)
obese_and_smoker=np.logical_and(df['smoker']==1, df['bmi']>30)
print("number of normal: ", normal.sum())
print("number of overweight: ", overweight.sum())
print("number of obese: ", obese.sum())
print("most unhealthy: ", obese_and_smoker.sum())
print(df.corr())#charges and smoker has a high positive correlation here as seen.
from sklearn.ensemble import RandomForestRegressor

rfc=RandomForestRegressor(n_estimators=15)

trainx=df[[ 'bmi', 'age']].iloc[:1070].values
trainy=df[['charges']].iloc[:1070].values
testx=df[['bmi', 'age']].iloc[1070:]
testy=df[['charges']].iloc[1070:]

print(trainx.shape, trainy.shape,  testy.shape, testx.shape)

rfc.fit(trainx, trainy)
pred_cost=rfc.predict(testx)

print(rfc.score(trainx, trainy))

X=testy
Y=pred_cost
Z=df['bmi'].iloc[1070:]
T=df['age'].iloc[1070:]
plt.subplots(figsize=(20, 10))
plt.scatter(Z, X, color='red')#real costs
plt.plot(Z, Y, color='blue', marker='*', linewidth=0)#predicted costs
plt.xlabel("bmi")
plt.ylabel("cost")
plt.title("Comparison of prediction and real values on the graph")
plt.show()

fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs=df['bmi'].iloc[1070:]#pred_cost
    ys=df['age'].iloc[1070:]
    zs=pred_cost
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('bmi')
ax.set_ylabel('age')
ax.set_zlabel('predicted costs')

plt.show()
