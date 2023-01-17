from IPython.display import Image
Image("../input/fotos1/contarside2.png")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # interpolation 

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../input/datas1/LoadForce1.txt",sep=",",header=None,names=["x","y","z","fx","fy","fz"])
distance = pd.read_csv("../input/datas1/LoadForce1.txt",header=None,names =["z-axis"])
z = distance.values

data[0:10]
fz =data.loc[:,"fz"].values.reshape(660,66) # 43560 data reshape to 660,66.
fx =data.loc[:,"fx"].values.reshape(660,66) # 43560 data reshape to 660,66

first_index = 48 # The fist atom of MoS2 Structure
second_index = 66 # The last atom of MoS2 Structure

new_fz= fz[:,first_index:second_index] # Pick only MoS2 structure from data
new_fx= fx[:,first_index:second_index] # Pick only MoS2 structure from data


sum_fx=[] # Summation all fx on the Mos2 
for i in range(0,660):
    sum_fx.append(new_fx[i].sum(axis=0))

sum_fz=[] #Summation all fz on the Mos2 
for i in range(0,660):
    sum_fz.append(new_fz[i].sum(axis=0))


sum_fz=np.array(sum_fz).reshape((44,15)) # Reshape data 44,15. 44 represent x-axis movement, 15 represent z-axis movement.
sum_fx=np.array(sum_fx).reshape((44,15)) # Reshape data 44,15. 44 represent x-axis movement, 15 represent z-axis movement.


z = pd.DataFrame(z)
z = [i[0] for i in z.values.tolist()]
for i in range (0,44):
    fz=(list(sum_fz[i]))
    #print(fz)
    #plt.plot(z,fz, color="blue", linestyle='dashed', linewidth=5, marker="o",markerfacecolor='red',markersize=12)
    #plt.xlabel('Z-distance(Ang)')
    #plt.ylabel('Force (Ev/Ang)')
    #plt.show()
    inter_point=10 # load 10
    #f1=interp1d(fz,z)
    #print(f1)
    
    interp_result_10=[]
for i in range (0,44):
    fx=list(sum_fx[i])
    #print(fx)
    #plt.plot(z,fx)
    #plt.show()
    #f2=interp1d(z,fx)
    #interp_result_10.append(f2(f1(inter_point))) # Create txt file with 10 eV/A load result
load_10 = pd.read_csv("../input/load10/interp_result_10.txt",header=None,names=["Laod"])
x_distance = pd.read_csv("../input/xdistance/xdistance.txt",header=None,names=["Seperation"])

print(load_10) # Load values 
l_10=load_10.values
x=x_distance.values # x distance value for sliding process
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

x = sc_X.fit_transform(x)
l_10 = sc_y.fit_transform(l_10)

print("load:",l_10[0:5])
print("x axis:",x[0:5])
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,l_10)

y_head = lr.predict(x)


from sklearn.preprocessing import PolynomialFeatures
py= PolynomialFeatures(degree=41)
x_pol = py.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_pol,l_10)

y_head2=lr2.predict(x_pol)

plt.plot(x,l_10,color="green", linestyle='dashed', linewidth=5, marker="o",markerfacecolor='blue',markersize=12)
plt.plot(x,y_head2,color='red',label="poly")
plt.xlabel('Displacement(Å)',fontsize=20)
plt.ylabel('Force (eV/Å)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend()
plt.show()

from sklearn.metrics import r2_score
print("linear regression score",r2_score(l_10,y_head))
print("Polynomial regression score",r2_score(l_10,y_head2))


#Prediction test
data = 0.49856467
data = np.asarray(data)
x_1 = py.fit_transform(data.reshape(-1,1)) 
y_head3=lr2.predict(x_1)
print("Prediction of polynomial regression",y_head3)
#print("coefficient of polynomial equations:",lr2.coef_)

#simulation data
y=-0.050488005681896 #interp_result_10[3]
print("Simulation Data:",y)
#Error Calculation
ec= abs(((y-y_head3)/y_head3)*100)
print("Error:",  ec)
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=5.215)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=5)
y_rbf = svr_rbf.fit(x,l_10).predict(x)
y_lin = svr_lin.fit(x, l_10).predict(x)
y_poly = svr_poly.fit(x,l_10 ).predict(x)
lw = 2
plt.scatter(x,l_10, color='red', label='data')
plt.plot(x, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(x, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(x, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('Displacement(Å)',fontsize=20)
plt.ylabel('Force (eV/Å)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Support Vector Regression')
plt.legend()
plt.show()
print("linear regression score",r2_score(l_10,y_lin))
print("Polynomial regression score",r2_score(l_10,y_poly))
print("SVM_rbf regression score",r2_score(l_10,y_rbf))
y_head4 = sc_y.inverse_transform(svr_rbf.predict(sc_X.transform(np.array([[0.62356467]]))))
print("SVR_rbf prediction :",y_head4)
#simulation data
y=-0.05776852172119981 #interp_result_10[8]
print("Simulation Data:",y)

#Error Calculation
ec= abs(((y_head4-y)/y)*100)
print("Error:", ec)
# Test data for prediction
o = 0.42356467
x_test = np.arange(o,2.02,0.025).reshape(-1,1)

y_head5 = sc_y.inverse_transform(svr_rbf.predict(sc_X.transform(x_test)))
print(y_head5.shape)
print(y_head5)

sc_X = StandardScaler()
x_test = sc_X.fit_transform(x_test)

plt.scatter(x_test,y_head5, color='red', label='Load')
plt.xlabel('Displacement(Å)',fontsize=20)
plt.ylabel('Force (eV/Å)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Test Data Prediction')
plt.legend()
plt.show()
data = pd.read_csv("../input/datas1/LoadForce1.txt",header=None,names=["x","y","z","fx","fy","fz"])
print(data.head())
# now we will shape our data to 660,66. Because we have 66 atoms in the structure. 
x = data.loc[:,"x"].values.reshape(660,66)
y = data.loc[:,"y"].values.reshape(660,66)
z = data.loc[:,"z"].values.reshape(660,66)
fx = data.loc[:,"fx"].values.reshape(660,66)
fy = data.loc[:,"fy"].values.reshape(660,66)
fz = data.loc[:,"fz"].values.reshape(660,66)
# Create list of x and z axis of MoS2's position.
atoms_x_position=[]
atoms_z_position=[]
for i in range(0,660):
    ax=list(x[i])
    az=list(z[i])
    atoms_x_position.append(ax)
    atoms_z_position.append(az)
    
atom_index = 55 # The 55th atom in structure. Sulfur atom.  
x_position=[] # Emtpy list for x-axis
z_position=[] # Emtpy list for z-axis
for i in range(0,660):
    ax1=(atoms_x_position[i][atom_index])
    az1=atoms_z_position[i][atom_index]
    x_position.append(ax1)
    z_position.append(az1)

sum_fx=[] # Emtpy list for Summation of Forces on the x-axis
for i in range(0,660):
    sum_fx.append(fx[i].sum(axis=0))

sum_fz=[] # Emtpy list for Summation of Forces on the z-axis
for i in range(0,660):
    sum_fz.append(fz[i].sum(axis=0))

sum_fy=[] # Emtpy list for Summation of Forces on the y-axis
for i in range(0,660):
    sum_fy.append(fy[i].sum(axis=0))
    
print(np.asarray(x_position).shape)
print(np.asarray(sum_fz).shape)
print(np.asarray(z_position).shape)
#Visualization

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(x_position[0:44],sum_fz[0:44],color="red")
ax1.set_xlabel("x position(Å)",fontsize=15)
ax1.set_ylabel("Total Force on z axis (Ev/Å)",fontsize=15)
ax2.plot(z_position[0:15],sum_fz[0:15],color="darkorange")
ax2.set_xlabel("z position(Å)",fontsize=15)
ax2.set_ylabel("Total Force on z axis(Ev/Å)",fontsize=15)
fig.tight_layout()
plt.subplots_adjust(right=2)
plt.show()
# Arrange data to pandas file
data1={"x": x_position,"z": z_position,"total fx": sum_fx,"total fy" :sum_fy,"total fz": sum_fz}
data2=pd.DataFrame(data1)

f1 = data2.iloc[:,[0,1]].values # Take the positions from data2
f2 = data2.loc[:,"total fz"].values.reshape(-1,1) 

#Normalization 
f2 = (f2-min(f2))/(max(f2)-min(f2))

print("Total Force on the z-axis:",f2[0])
print("Structure positions on x-axis and z-axis:",f1[2])

print("Shape of Positions:",f1.shape)
print("Shape of Total Force on z axis:",f2.shape)


#LabelEncoder use to convert our data into numbers, which our predictive models can better understand.
from sklearn import preprocessing

lab_enc = preprocessing.LabelEncoder()
f2 = lab_enc.fit_transform(f2)
f2=f2.reshape(-1,1)
print("Total Force on z axis:",f2[0:2])
from sklearn.linear_model import LogisticRegression
Mlr = LogisticRegression()
Mlr.fit(f1,f2)

#print(Mlr.intercept_)
#print(Mlr.coef_)
print(f2.shape)
print("Prediction with x-axis and z-axis",Mlr.predict(np.array([[2.35054,17.34373],[ 2.90252,17.44147]])))

print("Transform labelencoder data:",lab_enc.inverse_transform([387,234]))
y_headz = Mlr.predict(np.array(f1)).reshape(-1,1) #Prediction of total fz
pdata=lab_enc.inverse_transform(y_headz[3]).reshape(-1,1) # Prediction value Data
tdata=lab_enc.inverse_transform(f2[3]).reshape(-1,1) # True value Data
print("Prediction Value Data:",pdata)
print("True Value Data",tdata)

# Error calculation
ec = abs(((tdata-pdata)/pdata)*100)
print("Error:",ec)

f1= data2.loc[:,"x"].values.reshape(-1,1) # Take x position from data2
f2= data2.loc[:,"z"].values.reshape(-1,1) # Take z position from data2
f3 = data2.loc[:,"total fz"].values.reshape(-1,1) 

#Normalization 
f3 = (f3-min(f3))/(max(f3)-min(f3))
print(f3[0:3])
# Create one position data from multiply x position and z position
fnew=f1*f2
print("Shape of new position Data:",fnew.shape)
#Visualization
i=15 # Structure sliding process repeat the movement after 15 move on the z position but different x axis
fig,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(fnew[0:i],f3[0:i],color="red")
ax1.set_xlabel(" Position(Å)",fontsize=15)
ax1.set_ylabel("Total Force on z axis (Ev/Å)",fontsize=15)
ax2.plot(fnew[i:i*2],f3[i:i*2],color="darkorange")
ax2.set_xlabel(" Position(Å)",fontsize=15)
ax2.set_ylabel("Total Force on z axis(Ev/Å)",fontsize=15)
fig.tight_layout()
plt.subplots_adjust(right=2)
plt.show()
from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor(n_estimators=250,random_state=42)
rf.fit(fnew,f3)

y_head = rf.predict(fnew)
## R-square
print("Score of Random Forest :",r2_score(f3,y_head))
data = pd.read_csv("../input/datas1/LoadForce1.txt",sep=",",header=None,names=["x","y","z","fx","fy","fz"])

#print(data)
x = data.loc[:,"x"].values.reshape(660,66)
y = data.loc[:,"y"].values.reshape(660,66)
z = data.loc[:,"z"].values.reshape(660,66)

print("all data",x[0])


fx =data.loc[:,"fx"].values.reshape(660,66)
fy =data.loc[:,"fy"].values.reshape(660,66)
fz =data.loc[:,"fz"].values.reshape(660,66)
#Gold data
index_g1=0
index_g2=48

x_g = x[:,index_g1:index_g2]
y_g = y[:,index_g1:index_g2]
z_g = z[:,index_g1:index_g2]
fx_g = fx[:,index_g1:index_g2]
fy_g = fy[:,index_g1:index_g2]
fz_g = fz[:,index_g1:index_g2]
#MoS2 data
index_ms1=48
index_ms2=66

x_ms=x[:,index_ms1:index_ms2]
y_ms=y[:,index_ms1:index_ms2]
z_ms=z[:,index_ms1:index_ms2]
fx_ms=fx[:,index_ms1:index_ms2]
fy_ms=fy[:,index_ms1:index_ms2]
fz_ms=z[:,index_ms1:index_ms2]
x_g1=[]
y_g1=[]
z_g1=[]
fx_g1=[]
fy_g1=[]
fz_g1=[]

x_ms1=[]
y_ms1=[]
z_ms1=[]
fx_ms1=[]
fy_ms1=[]
fz_ms1=[]

for i in range(0,660):
    x_g1.append(x_g[i])
    y_g1.append(y_g[i])
    z_g1.append(z_g[i])
    fx_g1.append(fx_g[i])
    fy_g1.append(fy_g[i])
    fz_g1.append(fz_g[i])

    x_ms1.append(x_ms[i])
    y_ms1.append(y_ms[i])
    z_ms1.append(z_ms[i])
    fx_ms1.append(fx_ms[i])
    fy_ms1.append(fy_ms[i])
    fz_ms1.append(fz_ms[i])

x_g1=np.asarray(x_g1).reshape(31680,1)
y_g1=np.asarray(y_g1).reshape(31680,1)
z_g1=np.asarray(z_g1).reshape(31680,1)
fx_g1=np.asarray(fx_g1).reshape(31680,1)
fy_g1=np.asarray(fy_g1).reshape(31680,1)
fz_g1=np.asarray(fz_g1).reshape(31680,1)

x_ms1=np.asarray(x_ms1).reshape(11880,1)
y_ms1=np.asarray(y_ms1).reshape(11880,1)
z_ms1=np.asarray(z_ms1).reshape(11880,1)
fx_ms1=np.asarray(fx_ms1).reshape(11880,1)
fy_ms1=np.asarray(fy_ms1).reshape(11880,1)
fz_ms1=np.asarray(fz_ms1).reshape(11880,1)
gold=[x_g1,y_g1,z_g1,fx_g1,fy_g1,fz_g1]
gold = np.concatenate(gold,axis=1)
g_data=pd.DataFrame(gold) #,columns=["x","y","z","fx","fy","fz"]
gold_data= pd.read_csv("../input/gold11/Gold.csv",header=None,names=["x","y","z","fx","fy","fz"],index_col=None)
print(gold_data.head())
mos2 = [x_ms1,y_ms1,z_ms1,fx_ms1,fy_ms1,fz_ms1]
mos2 = np.concatenate(mos2,axis=1)
ms_data=pd.DataFrame(mos2)
#ms_data.to_csv("Mos2.csv")

mos2_data = pd.read_csv("../input/mos222/Mos2.csv",header=None,names=["x","y","z","fx","fy","fz"])
print(mos2_data.head())
gold_data["Atom"] = "Gold"
mos2_data["Atom"] = "MoS2"

data_AwM = pd.concat([gold_data,mos2_data],ignore_index=True)
#List Of Compensation
data_AwM["Atom"] =[1 if each == "Gold" else 0 for each in data_AwM["Atom"]]

print(data_AwM.head())
#Features and Class
x = data_AwM.drop(["Atom"],axis=1)
y = data_AwM["Atom"].values

x=np.nan_to_num(x)
y=np.nan_to_num(y)
#normalization
x = ((x-np.min(x))/np.max(x)-np.min(x))

#Train Test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
#logistic regression classification
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("logistic regression score:",lr.score(x_test,y_test))
print(lr.predict(x_test))

from sklearn.ensemble import RandomForestClassifier
dtc = RandomForestClassifier(n_estimators=5,random_state=42)
dtc.fit(x_train,y_train)
print("Random Forest Score:",dtc.score(x_test,y_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test,y_pred=dtc.predict(x_test))
#Vizualization
import seaborn as sns
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y Prediction")
plt.ylabel("y True")
plt.show()