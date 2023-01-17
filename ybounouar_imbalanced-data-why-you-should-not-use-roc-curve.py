import pandas as pd

import xgboost as xgb

import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_auc_score

import seaborn as sns

import matplotlib.pyplot as plt
def PlotConfusionMatrix(y_test,pred,y_test_legit,y_test_fraud):



    cfn_matrix = confusion_matrix(y_test,pred)

    cfn_norm_matrix = np.array([[1.0 / y_test_legit,1.0/y_test_legit],[1.0/y_test_fraud,1.0/y_test_fraud]])

    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix



    fig = plt.figure(figsize=(15,5))

    ax = fig.add_subplot(1,2,1)

    sns.heatmap(cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)

    plt.title('Confusion Matrix')

    plt.ylabel('Real Classes')

    plt.xlabel('Predicted Classes')



    ax = fig.add_subplot(1,2,2)

    sns.heatmap(norm_cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)



    plt.title('Normalized Confusion Matrix')

    plt.ylabel('Real Classes')

    plt.xlabel('Predicted Classes')

    plt.show()

    

    print('---Classification Report---')

    print(classification_report(y_test,pred))

    

def AUC_plot(true,pred):

    fpr, tpr, threshold = roc_curve(true, pred)

    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
df=pd.DataFrame.from_csv("../input/creditcard.csv").reset_index()

df.head()
df['hour'] = df['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)

del df["Time"]
X=df.iloc[:,:-1]

Y=df["Class"]



accuracy=[]

for k in range(3):

 

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=k)



    model = xgb.XGBClassifier(objective="binary:logistic")

    model.fit(X_train, y_train, eval_metric ="auc")



    y_pred = model.predict(X_test)

    predictions = [value for value in y_pred]



    accuracy.append(roc_auc_score(y_test, predictions))

    

    PlotConfusionMatrix(y_test,y_pred,y_test.value_counts()[0],y_test.value_counts()[1])

    

print("AUC : " + str(np.mean(accuracy)))
y_pred = model.predict(X)

PlotConfusionMatrix(Y,y_pred,Y.value_counts()[0],Y.value_counts()[1])
AUC_plot(Y,y_pred)