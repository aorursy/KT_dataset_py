import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib notebook
%matplotlib inline

data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.info()
data.describe()
# finding the number of nans
for i in data.columns:
    print(f"{i} has {np.isnan(data[i]).sum()} nans")
# Finding the number of unique values
for i in data.columns:
    print(f"{i} has {len(np.unique(data[i]))} unique values")
data["Outcome_category"] = data["Outcome"].apply(str).replace({"1":"Diabetic","0":"Non Diabetic"})
sns.pairplot(data.drop("Outcome",axis=1),hue = "Outcome_category")
plt.figure()
plt.pie([i for i in data.groupby("Outcome_category")["Outcome_category"].count()],explode = [0.1,0],labels = ["Diabetic","Non Diabetic"],autopct='%1.1f%%',shadow=True)
plt.xlabel("% of instances : Diabetic vs Non Diabetic",size= 15)

axes = ["ax1","ax2","ax3","ax4","ax5","ax6","ax7","ax8"]
plt.figure(figsize = (10,10))
for index,i in enumerate([421,422,423,424,425,426,427,428]):
    plt.subplot(i)
    axes[index] = sns.violinplot(y= data[data.columns[index]],x=data["Outcome_category"])
    plt.xlabel(data.columns[index])
    plt.tight_layout(h_pad=2,w_pad=2)
plt.show()

axes = ["ax1","ax2","ax3","ax4","ax5","ax6","ax7","ax8"]
plt.figure(figsize = (10,10))
for index,i in enumerate([421,422,423,424,425,426,427,428]):
    plt.subplot(i)
    axes[index] = sns.barplot(y= data[data.columns[index]].apply(np.mean),x=data["Outcome_category"],ci = None)
    plt.xlabel(f"{data.columns[index]} mean comparisons")
    plt.tight_layout(h_pad=2,w_pad=2)
plt.show()
from sklearn.feature_selection import SelectKBest , chi2 , f_classif
from sklearn.ensemble import ExtraTreesClassifier
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot = True)
plt.title("CORRELATION MATRIX",pad = 6)
model = ExtraTreesClassifier()
model.fit(data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]],data["Outcome"])
plt.figure()
plt.barh(y = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],width = model.feature_importances_)
plt.title("% of importance to determine Target label")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler
x_data = data[["BMI","Glucose","Age","DiabetesPedigreeFunction","Pregnancies","Insulin","SkinThickness","BloodPressure"]]
y_data = data["Outcome"]
from sklearn.linear_model import LogisticRegression
X_train , X_test , y_train , y_test = train_test_split(x_data,y_data,random_state = 74,test_size = 0.20)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
LR = LogisticRegression(penalty="l2",C = 0.05).fit(X_train,y_train)
from sklearn.metrics import accuracy_score , recall_score , precision_score , precision_recall_curve , roc_auc_score , roc_curve,classification_report, confusion_matrix 
LR_predictions = LR.predict(X_test)
accuracy_score(y_test,LR_predictions)
recall_score(y_test,LR_predictions)
precision_score(y_test,LR_predictions)
roc_auc_score(y_test,LR.predict_proba(X_test)[0:,1])
print("Accuracy of Ridge Linear Regression Model :",accuracy_score(y_test,LR_predictions))
print(classification_report(y_test,LR_predictions))
plt.figure()
ax = plt.subplot()
sns.heatmap(confusion_matrix(y_test,LR_predictions),cbar = False,annot = True)
ax.set_xlabel("Predicted Values",labelpad = 10)
ax.set_ylabel("True Values")
ax.xaxis.set_ticklabels(["Not Diabetic","Diabetic"])
ax.yaxis.set_ticklabels(["Not Diabetic","Diabetic"])
ax.set_title("CONFUSION MATRIX")
LR_predict_prob = LR.predict_proba(X_test)
LR_predict_prob = LR_predict_prob[:,1]
fpr, tpr , threshold = roc_curve(y_test,LR_predict_prob)

plt.figure(figsize=(10,5))
plt.plot(fpr,tpr,linestyle = "--")
plt.fill_between(fpr,tpr,alpha = 0.1)
plt.title("ROC for Ridge Logistic Regression")
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
print("ROC_AUC Score for Ridge Regression model :",roc_auc_score(y_test,LR_predict_prob))

