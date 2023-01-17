import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
FCSV_TRAIN="../input/train.csv"
FCSV_TESTX="../input/test.csv"
FCSV_TESTY="../input/gender_submission.csv"
Y="Survived"
REMOVE=["PassengerId","Name","Sex","Embarked","Ticket","Cabin","Age"]
# Load training data from train.csv
data_train=pd.read_csv(FCSV_TRAIN)
data_train=pd.concat([data_train.drop(REMOVE,axis=1),pd.get_dummies(data_train['Sex']),      \
                                                     pd.get_dummies(data_train['Embarked'])],axis=1)
data_train=data_train.drop(['female'],axis=1)
data_train=data_train.drop(['C']     ,axis=1)
data_train=data_train.dropna()
# Load test data from test.csv and gender_submission.csv
data_testx=pd.read_csv(FCSV_TESTX)
data_testy=pd.read_csv(FCSV_TESTY)
data_test=pd.concat([data_testy,data_testx],axis=1)
data_test=pd.concat([data_test.drop(REMOVE,axis=1),pd.get_dummies(data_test['Sex']),      \
                                                   pd.get_dummies(data_test['Embarked'])],axis=1)
data_test=data_test.drop(['female'],axis=1)
data_test=data_test.drop(['C'],axis=1)
data_test=data_test.dropna()
# Incorporate x in training and test data 
data_all=pd.concat([data_train,data_test])

x_all=data_all.drop([Y],axis=1)
y_all=data_all[Y]
x_all_ave=x_all.mean(axis=0)
y_all_ave=y_all.mean(axis=0)
x_all_std=x_all.std(axis=0,ddof=1)
y_all_std=y_all.std(axis=0,ddof=1)
# Auto-scaling with mean=0 and var=1 for all x variables (case 1)
xs_all=(x_all-x_all_ave)/x_all_std
print(xs_all.mean(axis=0))
print(xs_all.std(axis=0,ddof=1))
# Auto-scaling with mean=0 and var=1 for four x variables (case 2)
xs_part_all=x_all
xs_part_all['Pclass']=(xs_part_all['Pclass']-xs_part_all['Pclass'].mean(axis=0))/xs_part_all['Pclass'].std(axis=0,ddof=1)
xs_part_all['Parch'] =(xs_part_all['Parch'] -xs_part_all['Parch'].mean(axis=0)) /xs_part_all['Parch'].std(axis=0,ddof=1)
xs_part_all['SibSp'] =(xs_part_all['SibSp'] -xs_part_all['SibSp'].mean(axis=0)) /xs_part_all['SibSp'].std(axis=0,ddof=1)
xs_part_all['Fare']  =(xs_part_all['Fare']  -xs_part_all['Fare'].mean(axis=0))  /xs_part_all['Fare'].std(axis=0,ddof=1)

xs_part=pd.concat([xs_part_all['Pclass'],xs_part_all['Parch'],xs_part_all['SibSp'],xs_part_all['Fare']],axis=1)
print(xs_part_all.mean(axis=0))
print(xs_part_all.std(axis=0,ddof=1))
# Done PCA for case 1 and calculate some quantities
pca_case1=PCA()
pca_case1.fit(xs_all)
contribution_ratios_case1=pca_case1.explained_variance_ratio_ 
cumulative_contribution_ratios_case1=np.cumsum(contribution_ratios_case1)
score_case1=pca_case1.transform(xs_all)

# Done PCA for case 2 and calculate some quantities
pca_case2=PCA()
pca_case2.fit(xs_part)
contribution_ratios_case2=pca_case2.explained_variance_ratio_ 
cumulative_contribution_ratios_case2=np.cumsum(contribution_ratios_case2)
score_case2=pca_case2.transform(xs_part)
# Show results and visualization of reduced principle component space 
# Show results principle componets
fig,axes=plt.subplots(ncols=2,figsize=(10,4))
axes[0].set_xlabel("Number of PCs")
axes[0].set_ylabel("Contribution ratio (blue), Cumulative contribution ratio (red)")
axes[0].bar(np.arange(1,len(contribution_ratios_case1)+1),contribution_ratios_case1,align="center")
axes[0].plot(np.arange(1,len(cumulative_contribution_ratios_case1)+1),cumulative_contribution_ratios_case1,"ro-")
axes[0].grid(linestyle='dashed')
axes[1].bar(np.arange(1,len(pca_case1.explained_variance_)+1),pca_case1.explained_variance_,align="center")
axes[1].set_xlabel("Number of PCs")
axes[1].set_ylabel("Eigenvalues")
axes[1].grid(linestyle='dashed')
plt.tight_layout()
plt.show()

# Plot the reduced PC space: PC_m vs PC_n (m<n)
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(10,7))
m,n=0,0
for i in range(4):  
   for j in range(4):  
      if j>i: 
         if n==3:
            m+=1 
            n =0 
            if m==2: 
               break                  
         axes[m,n].grid(True,linestyle='dashed')
         x_ax_name="PC_"+str(i+1)
         y_ax_name="PC_"+str(j+1)
         axes[m,n].set_xlim(-4,8)
         axes[m,n].set_ylim(-4,8)
         axes[m,n].set_xlabel(x_ax_name)
         axes[m,n].set_ylabel(y_ax_name)
         axes[m,n].set_aspect('equal')
         axes[m,n].scatter(score_case1[:,i],score_case1[:,j],s=10,c=y_all)
         n+=1
plt.tight_layout()
plt.show()
# The boxplot of each of principle components             
# Show results and visualization of reduced principle component space 
# Show results principle componets
fig,axes=plt.subplots(ncols=2,figsize=(10,4))
axes[0].set_xlabel("Number of PCs")
axes[0].set_ylabel("Contribution ratio (blue), Cumulative contribution ratio (red)")
axes[0].bar(np.arange(1,len(contribution_ratios_case2)+1),contribution_ratios_case2,align="center")
axes[0].plot(np.arange(1,len(cumulative_contribution_ratios_case2)+1),cumulative_contribution_ratios_case2,"ro-")
axes[0].grid(linestyle='dashed')
axes[1].bar(np.arange(1,len(pca_case2.explained_variance_)+1),pca_case2.explained_variance_,align="center")
axes[1].set_xlabel("Number of PCs")
axes[1].set_ylabel("Eigenvalues")
axes[1].grid(linestyle='dashed')
plt.tight_layout()
plt.show()

# Plot the reduced PC space: PC_m vs PC_n (m<n)
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(10,7))
m,n=0,0
for i in range(4):  
   for j in range(4):  
      if j>i: 
         if n==3:
            m+=1 
            n =0 
            if m==2: 
               break                  
         axes[m,n].grid(True,linestyle='dashed')
         x_ax_name="PC_"+str(i+1)
         y_ax_name="PC_"+str(j+1)
         axes[m,n].set_xlim(-4,8)
         axes[m,n].set_ylim(-4,8)
         axes[m,n].set_xlabel(x_ax_name)
         axes[m,n].set_ylabel(y_ax_name)
         axes[m,n].set_aspect('equal')
         axes[m,n].scatter(score_case2[:,i],score_case2[:,j],s=10,c=y_all)
         n+=1
plt.tight_layout()
plt.show()         