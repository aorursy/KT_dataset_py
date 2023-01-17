import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

sns.set(style='white', palette='deep')

width = 0.35

fontsize = 10

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Function

def autolabel_without_pct(rects,ax): #autolabel

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy = (rect.get_x() + rect.get_width()/2, height),

                    xytext= (0,3),

                    textcoords="offset points",

                    ha='center', va='bottom')

def autolabel_horizontal(rects,ax):

    """

    Attach a text label above each bar displaying its height

    """

    for rect in rects:

        width = rect.get_width()

        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height()/2.,

                '%.2f' % width,

                ha='center', va='center', color='white')    
#Importing dataset

df = pd.read_csv('/kaggle/input/2006-2010-brazilian-congressman-election/vote.csv')
#Looking for null values

null_values = (df.isnull().sum()/len(df))*100

null_values = pd.DataFrame(null_values, columns=['% of null values'])

null_values
#Splitting dataset by year

df_2006 = df[df['ano'] == 2006]

df_2010 = df[df['ano'] == 2010]

print('In 2006 had {} candidates and in 2010 had {} candidates'.format(len(df_2006),len(df_2010)))
#In 2006 and 2010, which was the political party with more candidates elected?

from collections import Counter

df_2006_win = df_2006[df_2006['situacao'] == np.unique(df['situacao'])[0]]

df_2010_win = df_2010[df_2010['situacao'] == np.unique(df['situacao'])[0]]



party_win_2006 = Counter(df_2006_win['partido'])

party_win_2010 = Counter(df_2010_win['partido'])



party_win_2006 = {k: v for k, v in sorted(party_win_2006.items(), key=lambda item: item[1], reverse=True)}

party_win_2010 = {k: v for k, v in sorted(party_win_2010.items(), key=lambda item: item[1], reverse=True)}



party_win_2006_labels = list(party_win_2006.keys())

party_win_2006_values = list(party_win_2006.values())

party_win_2010_labels = list(party_win_2010.keys())

party_win_2010_values = list(party_win_2010.values())



ind_2006 = np.arange(len(party_win_2006_labels))

ind_2010 = np.arange(len(party_win_2010_labels))

fig = plt.figure(figsize=(10,10))

fig.suptitle('Top 5 Political Party in 2006 and 2010 in Brazil', fontsize=15) 

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

for i in np.arange(0,5):

    rects1 = ax1.bar(party_win_2006_labels[i],party_win_2006_values[i], width=width, edgecolor='black')

    rects2 = ax2.bar(party_win_2010_labels[i],party_win_2010_values[i], width=width, edgecolor='black')

    autolabel_without_pct(rects1, ax1)

    autolabel_without_pct(rects2, ax2)

   

ax1.set_xlabel('Political Party', fontsize=fontsize)

ax2.set_xlabel('Political Party', fontsize=fontsize)

ax1.set_xticks(ind_2006[:5])

ax2.set_xticks(ind_2010[:5])

ax1.set_xticklabels(party_win_2006_labels, fontsize=fontsize)

ax2.set_xticklabels(party_win_2010_labels, fontsize=fontsize)

ax1.set_ylabel('Number of Candidate Elected', fontsize=fontsize)

ax2.set_yticklabels([])

ax1.set_ylim(0,95)

ax2.set_ylim(0,95)

ax1.set_title('Top 5 Political Party in 2006', fontsize=fontsize)

ax2.set_title('Top 5 Political Party in 2010', fontsize=fontsize)

ax1.grid(b=True, which= 'major', linestyle='--' )

ax2.grid(b=True,which='major', linestyle='--')
#In 2006 and 2010, which was the top 10 candidate that collected more resources and was elected?

df_2006_win.columns

for x,y,z in [df_2006_win[['nome', 'uf', 'partido']]]:

    print(df_2006_win[x].values + ' ' + '('+ df_2006_win[y].values+')' + ' ' + '('+df_2006_win[z].values+')')



df_2006_win['nome-uf-partido'] = df_2006_win['nome'].values + ' ' + '('+ df_2006_win['uf'].values+')' + ' ' + '('+df_2006_win['partido'].values+')'

df_2010_win['nome-uf-partido'] = df_2010_win['nome'].values + ' ' + '('+ df_2010_win['uf'].values+')' + ' ' + '('+df_2010_win['partido'].values+')'



df_2006_win_receita = df_2006_win[['nome-uf-partido','total_receita','total_despesa' ]].sort_values(by='total_receita', ascending=False)[:10]

df_2010_win_receita = df_2010_win[['nome-uf-partido','total_receita','total_despesa' ]].sort_values(by='total_receita', ascending=False)[:10]



ind=np.arange(len(df_2006_win_receita))

fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)

rects1 = ax1.barh(ind+width/2,df_2006_win_receita['total_receita'].values, width, edgecolor= 'black', label='Total Resources', align='center')

rects2 = ax1.barh(ind-width/2,df_2006_win_receita['total_despesa'].values, width, edgecolor='black', label='Total Expenses', align='center')

ax1.set_title('The top 10 candidate that collected more resources and was elected in 2006')

ax1.set_yticks(ind)

ax1.set_yticklabels(df_2006_win_receita['nome-uf-partido'].values)

ax1.legend(loc='best', frameon=False)

ax1.grid(b=True, which='major', linestyle='--')

ax1.set_ylabel('Candidate / State / Political Party')

ax1.set_xlabel('Amount of Money (R$) - Brazilian Currency')

ax1.tick_params(axis='y', labelsize=10, labelcolor='k', labelrotation=0)

autolabel_horizontal(rects1,ax1)

autolabel_horizontal(rects2,ax1)



rects3 = ax2.barh(ind+width/2,df_2010_win_receita['total_receita'].values, width, edgecolor= 'black', label='Total Resources', align='center')

rects4 = ax2.barh(ind-width/2,df_2010_win_receita['total_despesa'].values, width, edgecolor='black', label='Total Expenses', align='center')

ax2.set_title('The top 10 candidate that collected more resources and was elected in 2010')

ax2.set_yticks(ind)

ax2.set_yticklabels(df_2010_win_receita['nome-uf-partido'].values)

ax2.legend(loc='best', frameon=False)

ax2.grid(b=True, which='major', linestyle='--')

ax2.set_ylabel('Candidate / State / Political Party')

ax2.set_xlabel('Amount of Money (R$) - Brazilian Currency')

ax2.tick_params(axis='y', labelsize=10, labelcolor='k', labelrotation=0)

autolabel_horizontal(rects3,ax2)

autolabel_horizontal(rects4,ax2)

plt.tight_layout()
#In 2006 and 2010, what were the numbers of men and women elected?

df_2006_win.columns

np.unique(df_2006_win['sexo'])

men_2006 = np.sum([df_2006_win['sexo'].values[i] == np.unique(df_2006_win['sexo'])[1] for i in np.arange(len(df_2006_win))])

women_2006 = np.sum([df_2006_win['sexo'].values[i] == np.unique(df_2006_win['sexo'])[0] for i in np.arange(len(df_2006_win))])



np.unique(df_2010_win['sexo'])

men_2010 = np.sum([df_2010_win['sexo'].values[i] == np.unique(df_2010_win['sexo'])[1] for i in np.arange(len(df_2006_win))])

women_2010 = np.sum([df_2010_win['sexo'].values[i] == np.unique(df_2010_win['sexo'])[0] for i in np.arange(len(df_2006_win))])



ind = np.arange(len(np.unique(df_2006_win['sexo'])))

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)

rects1 = ax.bar(ind[0]+width/2, men_2006,width=width, edgecolor='black', color='black', label='Men')

rects2 = ax.bar(ind[0]-width/2, women_2006,width=width, edgecolor='black', color='gray', label='Women')

rects3 = ax.bar(ind[1]+width/2, men_2010,width=width, edgecolor='black',color='black')

rects4 = ax.bar(ind[1]-width/2, women_2010,width=width, edgecolor='black', color='gray' )

ax.set_xticks(ind)

ax.legend(loc='best', title='Genre')

ax.set_xticklabels(['2006','2010'])

ax.set_xlabel('Years')

ax.set_ylabel('Amount')

ax.set_title('Numbers of men and women elected')

ax.grid(b=True, which='major', linestyle='--')

autolabel_without_pct(rects1,ax)

autolabel_without_pct(rects2,ax)

autolabel_without_pct(rects3,ax)

autolabel_without_pct(rects4,ax)
#In the Brazilian election, when the number of donations is not equal to the number collected resource, it can be a fraud signal.

#In 2006 and 2010, what were the candidates that the number of donations it did not equal the collected resource?

df_2006_win.columns

for x,y in [df_2006_win[['quantidade_doacoes', 'quantidade_doadores']]]:

    df_2006_win['fraud']= df_2006_win[x] - df_2006_win[y]

df_2006_win.sort_values(by='fraud', ascending=False)[['nome-uf-partido', 'fraud']]  

    

for x,y in [df_2010_win[['quantidade_doacoes', 'quantidade_doadores']]]:

    df_2010_win['fraud']= df_2010_win[x] - df_2010_win[y]

df_2010_win.sort_values(by='fraud', ascending=False)[['nome-uf-partido', 'fraud']]  
#In 2006 and 2010, what were the educational level of candidates ellected?

df_2006_win.columns

grouped_2006 = df_2006_win.groupby('grau')['grau'].count().sort_values(ascending=False)

grouped_2010 = df_2010_win.groupby('grau')['grau'].count().sort_values(ascending=False)

grouped_2006, grouped_2010
#In 2006 and 2010, what were the civil status of candidates ellected?

df_2006_win.columns

grouped_2006 = df_2006_win.groupby('estado_civil')['estado_civil'].count().sort_values(ascending=False)

grouped_2010 =df_2010_win.groupby('estado_civil')['estado_civil'].count().sort_values(ascending=False)

grouped_2006, grouped_2010
#Feature engineering

df_feature = df.copy()

df_feature['fraud'] = df_feature['quantidade_doacoes']-df_feature['quantidade_doadores'] 

df_feature['situacao'] = df_feature['situacao'].apply(lambda x: 1 if x==np.unique(df_feature['situacao'].values)[0] else 0)
#Getting dummies features

df_feature = pd.get_dummies(df_feature.drop(['partido','nome','uf','cargo', 'ocupacao'], axis=1))
#Avoiding dummies trap

df_feature.columns

df_feature = df_feature.drop(['sexo_FEMININO','grau_LÊ E ESCREVE','estado_civil_VIÚVO(A)'], axis=1)
## Correlation with independent Variable (Note: Models like RF are not linear like these)

df2 = df_feature.drop(['situacao'], axis=1)

df2.corrwith(df_feature.situacao).plot.bar(

        figsize = (10, 10), title = "Correlation with Situacao", fontsize = 15,

        rot = 90, grid = True)

#Splitting dataset into X and y

df_feature.columns

X = df_feature.drop(['ano', 'sequencial_candidato', 'situacao'],axis=1)

y = df_feature['situacao'] 
#Splitting the Dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

X_train.shape

X_test.shape

y_train.shape

y_test.shape
#Feature scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)

X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
#### Model Building ####

### Comparing Models



## Logistic Regression

from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(random_state = 0, penalty = 'l2')

lr_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = lr_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



results = pd.DataFrame([['Logistic Regression (Lasso)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
## K-Nearest Neighbors (K-NN)

#Choosing the K value

error_rate= []

for i in range(1,40):

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

print(np.mean(error_rate))
from sklearn.neighbors import KNeighborsClassifier

kn_classifier = KNeighborsClassifier(n_neighbors=25, metric='minkowski', p= 2)

kn_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = kn_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['K-Nearest Neighbors (minkowski)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## SVM (Linear)

from sklearn.svm import SVC

svm_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)

svm_linear_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = svm_linear_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)

## SVM (rbf)

from sklearn.svm import SVC

svm_rbf_classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)

svm_rbf_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = svm_rbf_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Naive Bayes

from sklearn.naive_bayes import GaussianNB

gb_classifier = GaussianNB()

gb_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = gb_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Naive Bayes (Gaussian)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

dt_classifier.fit(X_train, y_train)



#Predicting the best set result

y_pred = dt_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Decision Tree', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 200,

                                    criterion = 'gini')

rf_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = rf_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Gini (n=200)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Ada Boosting

from sklearn.ensemble import AdaBoostClassifier

ad_classifier = AdaBoostClassifier()

ad_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = ad_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Ada Boosting', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
##Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gr_classifier = GradientBoostingClassifier()

gr_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = gr_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Gradient Boosting', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
##Xg Boosting

from xgboost import XGBClassifier

xg_classifier = XGBClassifier()

xg_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = xg_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Xg Boosting', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
##Ensemble Voting Classifier

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score

voting_classifier = VotingClassifier(estimators= [('lr', lr_classifier),

                                                  ('kn', kn_classifier),

                                                  ('svc_linear', svm_linear_classifier),

                                                  ('svc_rbf', svm_rbf_classifier),

                                                  ('gb', gb_classifier),

                                                  ('dt', dt_classifier),

                                                  ('rf', rf_classifier),

                                                  ('ad', ad_classifier),

                                                  ('gr', gr_classifier),

                                                  ('xg', xg_classifier),],

voting='soft')



for clf in (lr_classifier,kn_classifier,svm_linear_classifier,svm_rbf_classifier,

            gb_classifier, dt_classifier,rf_classifier, ad_classifier, gr_classifier, xg_classifier,

            voting_classifier):

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



# Predicting Test Set

y_pred = voting_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Ensemble Voting', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)  
#The Best Classifier

print('The best classifier is:')

print('{}'.format(results.sort_values(by='Accuracy',ascending=False).head(5)))
#Applying K-fold validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=rf_classifier, X=X_train, y=y_train,cv=10)

accuracies.mean()

accuracies.std()

print("Randon Forest Gini (n=200) Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
#Plotting Cumulative Accuracy Profile (CAP)

y_pred_proba = rf_classifier.predict_proba(X=X_test)

import matplotlib.pyplot as plt

from scipy import integrate

def capcurve(y_values, y_preds_proba):

    num_pos_obs = np.sum(y_values)

    num_count = len(y_values)

    rate_pos_obs = float(num_pos_obs) / float(num_count)

    ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})

    xx = np.arange(num_count) / float(num_count - 1)

    

    y_cap = np.c_[y_values,y_preds_proba]

    y_cap_df_s = pd.DataFrame(data=y_cap)

    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(level = y_cap_df_s.index.names, drop=True)

    

    print(y_cap_df_s.head(20))

    

    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)

    yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0

    

    percent = 0.5

    row_index = int(np.trunc(num_count * percent))

    

    val_y1 = yy[row_index]

    val_y2 = yy[row_index+1]

    if val_y1 == val_y2:

        val = val_y1*1.0

    else:

        val_x1 = xx[row_index]

        val_x2 = xx[row_index+1]

        val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)

    

    sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1

    sigma_model = integrate.simps(yy,xx)

    sigma_random = integrate.simps(xx,xx)

    

    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)

    

    fig, ax = plt.subplots(nrows = 1, ncols = 1)

    ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')

    ax.plot(xx,yy, color='red', label='User Model')

    ax.plot(xx,xx, color='blue', label='Random Model')

    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)

    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')

    

    plt.xlim(0, 1.02)

    plt.ylim(0, 1.25)

    plt.title("CAP Curve - a_r value ="+str(ar_value))

    plt.xlabel('% of the data')

    plt.ylabel('% of positive obs')

    plt.legend()

    



capcurve(y_test,y_pred_proba[:,1])
## EXTRA: Confusion Matrix

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, annot=True, fmt='g')

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) 
## Feature Selection

# Recursive Feature Elimination

from sklearn.feature_selection import RFE



# Model to Test

classifier = LogisticRegression(random_state=0)



# Select Best X Features

rfe = RFE(classifier, n_features_to_select=None)

rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes

print(rfe.support_)

print(rfe.ranking_)

X_train.columns[rfe.support_]
# Fitting Model to the Training Set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)
# Predicting Test Set

y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Logistic Regression (RFE)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
# Evaluating Results

#Making the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(data=cm, annot=True)
#Making the classification report

from sklearn.metrics import classification_report

cr = classification_report(y_test,y_pred)
# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier,

                             X = X_train[X_train.columns[rfe.support_]],

                             y = y_train, cv = 10)

print("Logistic Regression (RFE) Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
results
#The Best Classifier

print('The best classifier is:')

print('{}'.format(results.sort_values(by='Accuracy',ascending=False)))