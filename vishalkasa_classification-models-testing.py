import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
Loan_Train=pd.read_csv("../input/loan-prediction-data/train.csv")
Loan_Test=pd.read_csv("../input/loan-prediction-data/test.csv")
Loan_Train.head()
Loan_Test.head()
Loan_Train=Loan_Train.drop(['Loan_ID','CoapplicantIncome'],axis=1)
Loan_Test=Loan_Test.drop(['Loan_ID'],axis=1)
Loan_Train.info()
Loan_Test.info()
Loan_Train.isnull().sum()
Loan_Test.isnull().sum()
print(Loan_Train.Gender.value_counts())
print(Loan_Test.Gender.value_counts())
Loan_Train.Gender.fillna("Male",inplace=True)
Loan_Test.Gender.fillna("Male",inplace=True)
print(Loan_Train.Married.value_counts())
Loan_Train.Married.fillna("Yes",inplace=True)
print(Loan_Train.Dependents.value_counts())
print(Loan_Test.Dependents.value_counts())
Loan_Train.Dependents.fillna("0",inplace=True)
Loan_Test.Dependents.fillna("0",inplace=True)
Loan_Train.Self_Employed.value_counts()
Loan_Test.Self_Employed.value_counts()
Loan_Train.Self_Employed.fillna("No",inplace=True)
Loan_Test.Self_Employed.fillna("No",inplace=True)
Loan_Train.describe()
Loan_Test.describe()
Loan_Train.LoanAmount.fillna(146,inplace=True)
Loan_Test.LoanAmount.fillna(136,inplace=True)
Loan_Train.Loan_Amount_Term.fillna(342,inplace=True)
Loan_Test.Loan_Amount_Term.fillna(342,inplace=True)
Loan_Train.Credit_History.fillna(1,inplace=True)
Loan_Test.Credit_History.fillna(1,inplace=True)
Loan_Train.isnull().sum()
Loan_Test.isnull().sum()
plt.bar(Loan_Train.Gender.unique(),Loan_Train.Gender.value_counts())
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
plt.bar(Loan_Train.Married.unique(),Loan_Train.Married.value_counts())
plt.xlabel("Married")
plt.ylabel("Count")
plt.show()
plt.pie(Loan_Train.Married.value_counts(),shadow=True,autopct='%1.1f%%',radius=1,labels=["Yes","No"])
plt.title("Married Status",loc="center")
sns.catplot("Gender",col="Married",data=Loan_Train,kind="count",size=3)
sns.catplot("Gender",col="Loan_Status",data=Loan_Train,kind="count",height=3)
sns.catplot("Married",col="Loan_Status",data=Loan_Train,kind="count",height=3)
sns.catplot("Education",col="Loan_Status",data=Loan_Train,kind="count",height=3)
sns.catplot("Self_Employed",col="Loan_Status",data=Loan_Train,kind="count",height=3)
sns.catplot("Dependents",col="Loan_Status",data=Loan_Train,kind="count",height=3)
sns.catplot("Property_Area",col="Loan_Status",data=Loan_Train,kind="count",height=3)
Loan_Train=pd.get_dummies(Loan_Train,drop_first=True)
Loan_Test=pd.get_dummies(Loan_Test,drop_first=True)
Loan_Train
x=Loan_Train.iloc[:,0:-1]
y=Loan_Train.Loan_Status_Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3,random_state=0)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
random_forest = RandomForestClassifier(n_estimators=40)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest_test = round(random_forest.score(X_test, Y_test) * 100, 2)
reg = LogisticRegression(solver='lbfgs',max_iter=1000)
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
acc_log = round(reg.score(X_train, Y_train) * 100, 2)
acc_log_test = round(reg.score(X_test, Y_test) * 100, 2)
decision_tree = DecisionTreeClassifier(max_depth=5) 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree_test = round(decision_tree.score(X_test, Y_test) * 100, 2)
test=KNeighborsClassifier(n_neighbors=4)
test.fit(X_train,Y_train)
ypred=test.predict(X_test)
acc_Kneighbour=test.score(X_train, Y_train) * 100
acc_Kneighbour_test=test.score(X_test, Y_test) * 100
linear_svc = LinearSVC(random_state=1,max_iter=10000)
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc_test = round(linear_svc.score(X_test, Y_test) * 100, 2)
Model=["RandomForestClassifier","DecisionTreeClassifier","KNeighborsClassifier","LogisticRegression","SVM"]
Accuracy=[acc_random_forest,acc_decision_tree,acc_Kneighbour,acc_log,acc_linear_svc]
plt.barh(Model,Accuracy)
Model=["RandomForestClassifier","DecisionTreeClassifier","KNeighborsClassifier","LogisticRegression","SVM"]
Accuracy=[acc_random_forest_test,acc_decision_tree_test,acc_Kneighbour_test,acc_log_test,acc_linear_svc_test]
plt.barh(Model,Accuracy)
from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, Y_train, Y_prediction)
models
