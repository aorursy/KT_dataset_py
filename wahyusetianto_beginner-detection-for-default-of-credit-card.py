# Import Packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc, roc_curve
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
%matplotlib inline
# Load Dataset
data = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
data.head()
# Dataset Info
data.info()
# Checking Categorical data
for kolom in ['SEX','EDUCATION','MARRIAGE'] :
    print(data[kolom].value_counts())
    print()
banyak = data['default.payment.next.month'].value_counts()
plt.figure(figsize = (6,6))
plt.title('Default Credit Card Clients\n (Default = 1, Not Default = 0)')
sns.barplot(x = banyak.index, y = banyak.values)
for i, v in enumerate(banyak.values):
    plt.text(i-.09 , v + 150, str(v),fontsize=11)
plt.xlabel('default.payment.next.month')
plt.ylabel('Value Counts')
plt.show()
pd.crosstab(data['default.payment.next.month'],data['SEX']).plot(kind='bar')
plt.title('default.payment.next.month vs SEX')
plt.xlabel('default.payment.next.month')
plt.ylabel('SEX')
plt.legend(['Male', 'Female'])
plt.show()
pd.crosstab(data['default.payment.next.month'],data['EDUCATION']).plot(kind='bar', figsize=(6,6))
plt.title('default.payment.next.month vs EDUCATION')
plt.xlabel('default.payment.next.month')
plt.ylabel('SEX')
plt.legend(['Graduate school','University','High School', 'Others', 'Unknown', 'Unknown'])
plt.show()
pd.crosstab(data['default.payment.next.month'],data['MARRIAGE']).plot(kind='bar')
plt.title('default.payment.next.month vs MARRIAGE')
plt.xlabel('default.payment.next.month')
plt.ylabel('MARRIAGE')
plt.legend(['Unknown', 'Married', 'Single', 'Others'])
plt.show()
plt.figure(figsize = (6,6))
plt.title('Distribution of Limit Ballance')
data['LIMIT_BAL'].hist(color='navy',alpha=.5,bins=19)
plt.show()
#Limit Ballance vs Sex
fig = px.box(data, x="SEX", y="LIMIT_BAL", color="SEX",width=700, height=700)
fig.update_traces(quartilemethod="exclusive")
fig.update_layout(title="Limit Ballance vs Sex")
fig.show()
#Limit Ballance, Education, and Marriage
fig_1 = px.box(data, x="EDUCATION", y="LIMIT_BAL", color='MARRIAGE',width=700, height=700)
fig_1.update_traces(quartilemethod="exclusive")
fig_1.update_layout(title='Limit Ballance vs Education')
fig_1.show()
#Limit Ballance, Education, and Marriage
fig_2 = px.box(data, x="EDUCATION", y="LIMIT_BAL", color = 'SEX',width=700, height=700)
fig_2.update_traces(quartilemethod="exclusive")
fig_2.update_layout(title='Limit Ballance vs Education')
fig_2.show()
#Limit Ballance, Education, and Marriage
plt.figure(figsize = (7,7))
plt.title('Distribution of Age')
sns.distplot(data['AGE'])
plt.show()
#Limit Ballance, Age, and Default Payment Next Month
fig_3 = px.box(data, x="AGE", y="LIMIT_BAL", color='SEX')
fig_3.update_traces(quartilemethod="exclusive")
fig_3.update_layout(title='Limit Ballance vs Age')
fig_3.show()
#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data[['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'SEX', 'EDUCATION','MARRIAGE','default.payment.next.month']].corr(method="spearman"), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
#Labeling Columnsling Collumns
y = data['default.payment.next.month']
X = data[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE',
          'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
          'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
          'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
class predict(object):
    def __init__(self, X, Y):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X , Y, test_size=0.2)

    def Select_Algorithm(self):
        algo = {'Logistic' : {'model' : [LogisticRegression()]},
                'Random Forest' : {'model' : [RandomForestClassifier(n_estimators=500)]},
                'Naive Bayes' : {'model' : [GaussianNB()]},
                'AdaBoost' : {'model' : [AdaBoostClassifier(n_estimators=500)]},
                'Gradien' : {'model' : [GradientBoostingClassifier(n_estimators=500)]}
        }
        res = {'model' : '', 'accuracy' : 0, 'Matrix' : None}
        for x in algo:
            for model in algo[x]['model']:
                model.fit(self.X_train, self.Y_train)
                pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.Y_test, pred)
                if accuracy > res['accuracy']:
                    res['model'] = model
                    res['accuracy'] = accuracy
                    res['Matrix'] = confusion_matrix(self.Y_test, pred)
                print(x, accuracy)
        self.models = algo
        return res
base_model = predict(X, y)
model = base_model.Select_Algorithm()
res_model = model['model']
def Matrix(cnf_matrix):
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    return plt.show()

def plot_roc(y_test, y_pred):
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return plt.show()

def plot_Feature(feature, clf):
    tmp = pd.DataFrame({'Feature': feature, 
                        'Feature importance': clf.feature_importances_})
    tmp = tmp.sort_values(by='Feature importance',ascending=False)
    plt.figure(figsize = (7,4))
    plt.title('Features importance',fontsize=14)
    s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    return plt.show()
plot_Feature(X.columns, res_model)
Matrix(model['Matrix'])
plot_roc(res_model.predict(base_model.X_test), base_model.Y_test)