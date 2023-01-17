import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
#Importing Data

df = pd.read_csv("../input/heart.csv")
df.columns
#Improving Names

new_columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholestoral',

               'fasting_blood_sugar', 'resting_electrocardiographic_results', 

               'maximum_heart_rate', 'exercise_induced_angina', 'oldpeak', 'slope', 

               'ca','thal','target']
#Replacing Columns Names

for i in range(0, len(df.columns)):

    df.columns.values[i] = new_columns[i] 
df_user = pd.DataFrame(np.arange(0, len(df)),columns = ['patient'])

df = pd.concat([df_user,df], axis=1)
df.info()
df.head(5)
df.tail(5)
df.describe()
df['target'].value_counts()
#Verifying Null Values

sns.heatmap(data=df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df.isna().sum()
df.isna().any()
#Defyning X and y

X = df.drop(['patient','target'], axis=1)

y = df['target']
sns.set(style = 'ticks', color_codes=True)

sns.pairplot(data=df, hue='target',  

             vars=['resting_blood_pressure',

                                          'cholestoral',

                                          'maximum_heart_rate',

                                          'oldpeak', 'age'])
sns.countplot(data=df, x='target')
plt.figure(figsize=(15,10))

sns.heatmap(data= df.corr(), annot=True, cmap='viridis')
fig, ax = plt.subplots(4,2, figsize=(16,16))

sns.distplot(df.age, bins = 20, ax=ax[0,0]) 

sns.distplot(df.resting_blood_pressure, bins = 20, ax=ax[0,1]) 

sns.distplot(df.chest_pain_type, bins = 20, ax=ax[1,0]) 

sns.distplot(df.cholestoral, bins = 20, ax=ax[1,1]) 

sns.distplot(df.maximum_heart_rate, bins = 20, ax=ax[2,0])

sns.distplot(df.oldpeak, bins = 20, ax=ax[2,1])

sns.distplot(df.slope, bins = 20, ax=ax[3,0]) 

sns.distplot(df.thal, bins = 20, ax=ax[3,1]) 
df2 = df[['resting_blood_pressure','cholestoral','maximum_heart_rate','oldpeak']]

fig = plt.figure(figsize=(15, 12))

plt.suptitle('Histograms of Numerical Columns', fontsize=20)

for i in range(df2.shape[1]):

    plt.subplot(4, 2, i + 1)

    f = plt.gca()

    f.set_title(df2.columns.values[i])



    vals = np.size(df2.iloc[:, i].unique())

    if vals >= 100:

        vals = 100

    

    plt.hist(df2.iloc[:, i], bins=vals, color='#3F5D7D')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    
#Correlation with independent variable

X.corrwith(df.target).plot.bar(figsize = (15, 10), title = "Correlation with Target", fontsize = 10,grid = True)
## Correlation Matrix

sns.set(style="white")



# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



## Pie Plots 

df2 = df[['sex', 'chest_pain_type', 'fasting_blood_sugar',

                    'resting_electrocardiographic_results', 'exercise_induced_angina', 'slope',

                    'ca', 'thal', 'target',

                                        ]]

fig = plt.figure(figsize=(20, 15))

plt.suptitle('Pie Chart Distributions', fontsize=20)

for i in range(1, df2.shape[1] + 1):

    plt.subplot(6, 3, i)

    f = plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(df2.columns.values[i - 1])

   

    values = df2.iloc[:, i - 1].value_counts(normalize = True).values

    index = df2.iloc[:, i - 1].value_counts(normalize = True).index

    plt.pie(values, labels = index, autopct='%1.1f%%')

    plt.axis('equal')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#Splitting the dataset into the traing set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=0)
#Feature scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))

X_test2 = pd.DataFrame(sc_X.transform(X_test))

X_train2.columns = X_train.columns.values

X_test2.columns = X_test.columns.values

X_train2.index = X_train.index.values

X_test2.index = X_test.index.values

X_train = X_train2

X_test = X_test2

X_train.head(5)
X_test.head(5)
#Model Building (Logistic Regression)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0, penalty='l1')

classifier.fit(X_train, y_train)
#Predicting Test set

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score, confusion_matrix

acc = accuracy_score(y_test,y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test,y_pred)

results = pd.DataFrame([['Logistic Regression (Lasso)', acc,prec,rec,f1]],columns=['Model', 'Accuracy', 'Precision', 'Recall','F1 Score'])

results
## EXTRA: Confusion Matrix

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, annot=True, fmt='g')

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) 
#Plotting Cumulative Accuracy Profile (CAP)

y_pred_prob = classifier.predict_proba(X_test)

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

    
capcurve(y_test,y_pred_prob[:,1])
#Applying K-folds validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator= classifier, X=X_train, y=y_train, cv=10)

accuracies.mean()

accuracies.std()

print('Logistic Regression (Lasso) Accuracy: %0.3f (+/- %0.3f)' % (accuracies.mean(), accuracies.std() * 2))
#Analyzing the coefficients

pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]),

           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])

           ],axis = 1)
#Feature selection 

#Recursive feature elimination

from sklearn.feature_selection import RFE



#Select best feature 

rfe = RFE(classifier, n_features_to_select= None)

rfe = rfe.fit(X_train, y_train)



#Summarize the selection of the attributes

print(rfe.support_)

print(rfe.ranking_)

X_train.columns[rfe.support_]



# New Correlation Matrix

sns.set(style="white")



# Compute the correlation matrix

corr = X_train[X_train.columns[rfe.support_]].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})  
# Fitting Model to the Training Set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0, penalty= 'l1')

classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Logistic Regression RFE (Lasso)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)

results
y_pred_prob = classifier.predict_proba(X_test[X_train.columns[rfe.support_]])

capcurve(y_test,y_pred_prob[:,1])