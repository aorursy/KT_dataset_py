import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
#Naming the columns
Headers=['Age','Sex','Chest_Pain','Resting_Blood_Pressure','Colestrol','Fasting_Blood_Sugar','Rest_ECG','MAX_Heart_Rate','Exercised_Induced_Angina','ST_Depression','Slope','Major_Vessels','Thalessemia','Target']
df1=pd.read_csv('../input/heart-diseases/Heart_Disease.csv',names=Headers)
df1.head()
df1.head()
df1.dtypes
df1.rename(columns={'cp':'Chest Pain','trestbps':'Resting Blood Pressure','fbs':'Fasting Blood Sugar','thalach':'Maximum Heart Rate Achieved','exang':'Exercise Induced Angina','ca':'Number of Major Vessels'},inplace=True)
df1.head()
df1.info()
df=df1[(df1['Thalessemia']!='?') &(df1['Major_Vessels']!='?')]
df['Thalessemia']=pd.to_numeric(df['Thalessemia'])
df['Major_Vessels']=pd.to_numeric(df['Major_Vessels'])
df.describe()
#Mapping the numerical attributes into categorical attributes
df['Gender']=df['Sex'].map({1:'Male',0:'Female'})
df['Defect']=df['Thalessemia'].map({3 :'normal', 6 : 'fixed defect', 7 : 'reversable defect'})
df['ChestPain']=df['Chest_Pain'].map({1: 'typical angina',  2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'})
df['FastingBloodSugar']=df['Fasting_Blood_Sugar'].map({1 : 'true', 0 : 'false'})
df['RestECG']=df['Rest_ECG'].map({0 : 'normal', 1 : 'having ST-T wave abnormality', 2 : 'showing probable or definite left ventricular hypertrophy by Estes criteria'})
df['exercisedinducedangina']=df['Exercised_Induced_Angina'].map({1:'yes',0:'no'})
df['slope']=df['Slope'].map({ 1:'upsloping',  2: 'flat', 3: 'downsloping'})
df['Target']=df['Target'].map({0:0,1:1,2:1,3:1})
df["Target"].replace(np.nan, 0, inplace=True)
df.head()
df.columns
corr=df[['Age', 'Sex', 'Chest_Pain', 'Resting_Blood_Pressure', 'Colestrol','Fasting_Blood_Sugar', 'Rest_ECG', 'MAX_Heart_Rate','Exercised_Induced_Angina', 'ST_Depression', 'Slope', 'Major_Vessels','Thalessemia', 'Target']].corr()
corr.head()
#plotting the correlation matrix
plt.figure(figsize=(15,10))
sns.heatmap(corr,vmax=.3, square=True,annot=True)

pearson_coef, p_value = stats.pearsonr(df['Age'], df['Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
count, bin_edges = np.histogram(df['Age'])

df['Age'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
plt.title("Age Counts")
plt.xlabel("Age Range")
plt.ylabel("Count")

plt.show()
pearson_coef, p_value = stats.pearsonr(df['Sex'], df['Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
ax=sns.barplot(data=df,x="Sex",y="Target")
ax.set_title("Sex vs Target")
ax.set_xlabel("Gender")
ax.set_ylabel("Target Count")
plt.show()
df.Target.value_counts()
dfsvt=df[['Sex','Target','Age']]
SVT=dfsvt.groupby(['Sex','Target'],as_index=False).count()
SVT.rename(columns={'Age':'Count'},inplace=True)
SVT

pd.pivot_table(SVT, index = 'Sex', columns='Target',values='Count')
sns.barplot(data=SVT, x="Sex", y = "Count", hue="Target")
plt.show()
pearson_coef, p_value = stats.pearsonr(df['Chest_Pain'], df['Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
ax=sns.violinplot(data=df,x="Chest_Pain",y="Target")
ax.set_title("Chest Pain vs Target")
ax.set_xticklabels(['Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic'])
ax.set_xlabel("Chest Pain")
ax.set_ylabel("Target Average")
plt.show()
ax1=sns.barplot(data=df,x="Chest_Pain",y="Target")
ax1.set_title("Chest Pain vs Target")
ax1.set_xticklabels(['Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic'])
ax1.set_xlabel("Chest Pain")
ax1.set_ylabel("Target Average")
plt.show()
df.Chest_Pain.value_counts()
dfcpvt=df[['Chest_Pain','Target','Age']]
CPVT=dfcpvt.groupby(['Chest_Pain','Target'],as_index=False).count()
CPVT.rename(columns={'Age':'Count'},inplace=True)
CPVT

pd.pivot_table(CPVT, index = 'Chest_Pain', columns='Target',values='Count')
sns.barplot(data=CPVT, x="Chest_Pain", y = "Count", hue="Target")
plt.show()
pearson_coef, p_value = stats.pearsonr(df['MAX_Heart_Rate'], df['Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
count, bin_edges = np.histogram(df['MAX_Heart_Rate'])

df['MAX_Heart_Rate'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
plt.title("Heart Rate Counts")
plt.xlabel("Heart Rate")
plt.ylabel("Counts")

plt.show()
pearson_coef, p_value = stats.pearsonr(df['Exercised_Induced_Angina'], df['Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
ax1=sns.barplot(data=df,x="Exercised_Induced_Angina",y="Target")
ax1.set_title("Exercicse vs Target")
ax1.set_xticklabels(['No','Yes'])
ax1.set_xlabel("Exericed Induced Angina")
ax1.set_ylabel("Target Average")
plt.show()
df.Exercised_Induced_Angina.value_counts()
dfexvt=df[['Exercised_Induced_Angina','Target','Age']]
EXVT=dfexvt.groupby(['Exercised_Induced_Angina','Target'],as_index=False).count()
EXVT.rename(columns={'Age':'Count'},inplace=True)
EXVT

pd.pivot_table(EXVT, index = 'Exercised_Induced_Angina', columns='Target',values='Count')
sns.barplot(data=EXVT, x="Exercised_Induced_Angina", y = "Count", hue="Target")
plt.show()
pearson_coef, p_value = stats.pearsonr(df['ST_Depression'], df['Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
count, bin_edges = np.histogram(df['ST_Depression'])

df['ST_Depression'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
plt.title("ST_Depression")
plt.xlabel("ST_Depression")
plt.ylabel("Counts")

plt.show()
pearson_coef, p_value = stats.pearsonr(df['Slope'], df['Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
ax1=sns.barplot(data=df,x="Slope",y="Target")
ax1.set_title("Slope vs Target")
ax1.set_xticklabels(['Upsloping','Flat','Downsloping'])
ax1.set_xlabel("Slope")
ax1.set_ylabel("Target Average")
plt.show()
df.Slope.value_counts()
dfslvt=df[['Slope','Target','Age']]
SLVT=dfslvt.groupby(['Slope','Target'],as_index=False).count()
SLVT.rename(columns={'Age':'Count'},inplace=True)
SLVT

pd.pivot_table(SLVT, index = 'Slope', columns='Target',values='Count')
sns.barplot(data=SLVT, x="Slope", y = "Count", hue="Target")
plt.show()
pearson_coef, p_value = stats.pearsonr(df['Thalessemia'], df['Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
ax1=sns.barplot(data=df,x="Thalessemia",y="Target")
ax1.set_title("Thalessemia vs Target")
ax1.set_xticklabels(['Normal','Fixed Defect','Reversible Defect'])
ax1.set_xlabel("Thalessemia")
ax1.set_ylabel("Target Average")
plt.show()
df.Thalessemia.value_counts()
dfthvt=df[['Thalessemia','Target','Age']]
THVT=dfthvt.groupby(['Thalessemia','Target'],as_index=False).count()
THVT.rename(columns={'Age':'Count'},inplace=True)
THVT

pd.pivot_table(THVT, index = 'Thalessemia', columns='Target',values='Count')
sns.barplot(data=THVT, x="Thalessemia", y = "Count", hue="Target")
plt.show()
pearson_coef, p_value = stats.pearsonr(df['Major_Vessels'], df['Target'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
ax1=sns.barplot(data=df,x="Major_Vessels",y="Target")
ax1.set_title("Major_Vessels vs Target")
ax1.set_xlabel("Major_Vessels")
ax1.set_ylabel("Target Average")
plt.show()
df.Major_Vessels.value_counts()
dfmvvt=df[['Major_Vessels','Target','Age']]
MVVT=dfmvvt.groupby(['Major_Vessels','Target'],as_index=False).count()
MVVT.rename(columns={'Age':'Count'},inplace=True)
MVVT

pd.pivot_table(MVVT, index = 'Major_Vessels', columns='Target',values='Count')
sns.barplot(data=MVVT, x="Major_Vessels", y = "Count", hue="Target")
plt.show()
df.drop(['Age','Resting_Blood_Pressure','Fasting_Blood_Sugar','Colestrol','Rest_ECG','Sex','Thalessemia','Chest_Pain','FastingBloodSugar','RestECG','Exercised_Induced_Angina','Slope'],axis=1,inplace=True)
df.head()
#doing one hot encoding
dff=pd.get_dummies(df,columns=['Gender','Defect','ChestPain','exercisedinducedangina','slope'])
dff.head()
from sklearn.metrics import classification_report, confusion_matrix
import itertools

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X=dff[['MAX_Heart_Rate', 'ST_Depression', 'Major_Vessels', 'Defect_fixed defect', 'Defect_normal','Defect_reversable defect', 'ChestPain_asymptomatic','ChestPain_atypical angina', 'ChestPain_non-anginal pain','ChestPain_typical angina', 'exercisedinducedangina_no','exercisedinducedangina_yes']]
Y=dff['Target']
train_data,test_data,train_labels,test_labels=train_test_split(X,Y,random_state=1,test_size=0.2)

tree= DecisionTreeClassifier(criterion="entropy",random_state=1,max_depth=3)
tree.fit(train_data,train_labels)
tree.score(test_data,test_labels)

prediction=tree.predict(test_data)
print(prediction[5:15])
print(test_labels[5:15])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Compute confusion matrix
cnf_matrix = confusion_matrix(test_labels, prediction, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(test_labels, prediction))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Healthy(0)','Sick(1)'],normalize= False,  title='Confusion matrix')
from sklearn.svm import SVC
classifier= SVC(kernel='rbf',C=100,gamma=100)
classifier.fit(train_data,train_labels)
svmpred=classifier.predict(test_data)
classifier.score(test_data,test_labels)
cnf_matrix = confusion_matrix(test_labels, svmpred, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(test_labels, svmpred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Healthy(0)','Sick(1)'],normalize= False,  title='Confusion matrix')
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(train_data,train_labels)
model.score(test_data,test_labels)


Logpred=model.predict(test_data)
print(prediction[5:15])
print(test_labels[5:15])
cnf_matrix = confusion_matrix(test_labels, Logpred, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(test_labels, Logpred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Healthy(0)','Sick(1)'],normalize= False,  title='Confusion matrix')
from sklearn.ensemble import RandomForestClassifier
rfclassifier=RandomForestClassifier(n_estimators=2000,random_state=0)
rfclassifier.fit(train_data,train_labels)
rfclassifier.score(test_data,test_labels)


rfpred=rfclassifier.predict(test_data)
cnf_matrix = confusion_matrix(test_labels, rfpred, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(test_labels, rfpred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Healthy(0)','Sick(1)'],normalize= False,  title='Confusion matrix')
from sklearn.neighbors import KNeighborsClassifier
knnclassifier=KNeighborsClassifier(n_neighbors=5)
knnclassifier.fit(train_data,train_labels)
knnclassifier.score(test_data,test_labels)

knnpred=knnclassifier.predict(test_data)
cnf_matrix = confusion_matrix(test_labels, knnpred, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(test_labels, knnpred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Healthy(0)','Sick(1)'],normalize= False,  title='Confusion matrix')
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
print(f1_score(test_labels,prediction))
print(jaccard_score(test_labels,prediction))
print(f1_score(test_labels,svmpred))
print(jaccard_score(test_labels,svmpred))
print(f1_score(test_labels,Logpred))
print(jaccard_score(test_labels,Logpred))
print(f1_score(test_labels,rfpred))
print(jaccard_score(test_labels,rfpred))
print(f1_score(test_labels,knnpred))
print(jaccard_score(test_labels,knnpred))

