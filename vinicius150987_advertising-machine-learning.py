import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style= 'darkgrid', palette='deep')

import warnings

warnings.filterwarnings('ignore')

bins = range(0,100,10)

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/advertising.csv')
df.head()
df.tail()
df.info()
df_feature = df.copy()
#Creating a user columns

df_user = pd.DataFrame(np.arange(0, len(df_feature)), columns=['user'])
df_feature = pd.concat([df_user, df_feature], axis=1)
df_feature.groupby('Country')['Country'].unique().sort_values()
#Removing parentheses from Country

def removeAfterParentheses(string):

    """

    input is a string 

    output is a string with everything after comma removed

    """

    return string.split('(')[0].strip()

df_feature.Country = df_feature.Country.apply(removeAfterParentheses)
#Checking the remove parentheses 

df_feature.groupby('Country')['Country'].unique().sort_values()
countries = df_feature.groupby('Country')['Country'].unique().sort_values()
#Installing country_converter package

!pip install country_converter --upgrade
#Extracting Countries continent

import country_converter as coco

cc = coco.CountryConverter()

continent = np.array([])

for i in range(0, len(df_feature)):

    continent= np.append(continent, cc.convert(names=df_feature['Country'][i], to='Continent' ))
df_feature['continent'] = pd.DataFrame(continent) 
df_feature.columns
#Reorganizing the columns

df_feature = df_feature[['user','Daily Time Spent on Site', 'Age', 'Area Income',

       'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country', 'continent',

       'Timestamp', 'Clicked on Ad']]
#Installing date_converter package

!pip install easy-date
#Converting string format to Datatime format 

import date_converter



for i in range(0,len(df_feature)):

    df_feature['Timestamp'][i] = date_converter.string_to_datetime(df_feature['Timestamp'][i], '%Y-%m-%d %H:%M:%S')

time_new = df_feature['Timestamp'].iloc[0]

df_feature['Hour'] = df_feature['Timestamp'].apply(lambda time_new: time_new.hour)

df_feature['Month'] = df_feature['Timestamp'].apply(lambda time_new: time_new.month)

df_feature['Day'] = df_feature['Timestamp'].apply(lambda time_new: time_new.weekday())
df_feature.info()
df_feature.head()
#How many percentage the user have been spending on site? Creating % spending time columns

df_feature.columns

df_feature['% spending time'] =  ((df_feature['Daily Time Spent on Site'] / df_feature['Daily Internet Usage']) * 100 )

df_feature = df_feature[['user', 'Daily Time Spent on Site','Daily Internet Usage',

                                        '% spending time','Age','Area Income',

                                        'Ad Topic Line', 'City', 'Male', 'Country',

                                        'continent', 'Timestamp','Hour', 'Month', 'Day','Clicked on Ad']]
df_feature.head()
def bar_chart(feature1, feature2):

    g = pd.crosstab(df_feature[feature1], df_feature[feature2]).plot(kind='bar', figsize=(10,10), rot = 45)

    ax = g.axes

    for p in ax.patches:

     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points') 

    plt.grid(b=True, which='major', linestyle='--')

    plt.legend(['Clicked on Ad',"Did not Clicked on Ad"])

    plt.title('Clicked on Ad for {}'.format(feature1))

    plt.xlabel('{}'.format(feature1))

    plt.tight_layout()

    plt.ylabel('Quantity')

    

def bar_chart_group(feature):

    g = pd.crosstab(pd.cut(df_feature[feature], bins), df_feature['Clicked on Ad']).plot(kind='bar', figsize=(10,10), rot = 45)

    ax = g.axes

    for p in ax.patches:

     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points') 

    plt.grid(b=True, which='major', linestyle='--')

    plt.legend(['Clicked on Ad',"Did not Clicked on Ad"])

    plt.title('Clicked on Ad for {}'.format(feature))

    plt.xlabel('{}'.format(feature))

    plt.tight_layout()

    plt.ylabel('Quantity')



def bar_chart_hour(feature):

    bins_hour = np.arange(0,25,12)

    g = pd.crosstab(pd.cut(df_feature[feature], bins_hour), df_feature['Clicked on Ad']).plot(kind='bar', figsize=(10,10), rot = 45)

    ax = g.axes

    for p in ax.patches:

     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points') 

    plt.grid(b=True, which='major', linestyle='--')

    plt.legend(['Clicked on Ad',"Did not Clicked on Ad"])

    plt.title('Clicked on Ad for {}'.format(feature))

    plt.xlabel('{}'.format(feature))

    plt.tight_layout()

    plt.ylabel('Quantity')
#Taking latitude and longitude

from geopy.geocoders import Nominatim

lat = np.array([])

lon = np.array([])

country = np.array([])



for i in range(0, len(countries)):

    geolocator = Nominatim(user_agent='tito', timeout=100)

    location = geolocator.geocode(countries.index[i], timeout=100)

    lat = np.append(lat, location.latitude)

    lon = np.append(lon, location.longitude)

    country = np.append(country, countries.index[i])
#Importing Map

import folium

data = pd.DataFrame({

'lat':lat,

'lon':lon,

'name':country})

data.head()    



m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2 , )

country_map = list(zip(data['name'].values, data['lat'].values, data['lon'].values))

# add features

for country_map in country_map:

    folium.Marker(

        location=[float(country_map[1]), float(country_map[2])],

        popup=folium.Popup(country_map[0], parse_html=True),

        icon=folium.Icon(icon='home')

    ).add_to(m)   

    

m  
bar_chart('Male','Clicked on Ad')

bar_chart('continent', 'Clicked on Ad')

bar_chart('Day', 'Clicked on Ad')

bar_chart('Month', 'Clicked on Ad')

bar_chart_group('Age')

bar_chart_group('% spending time')

bar_chart_hour('Hour')
df_feature.drop(['user', 'Male', 'Clicked on Ad'], axis=1).hist(figsize=(10,10))
df_feature.groupby('continent')['Area Income'].sum().sort_values().plot(kind='bar', figsize=(10,10), rot=45)

plt.title('Area income per Continent')

plt.grid(b=True, which='major', linestyle='--')

plt.tight_layout()

plt.ylabel('Quantity')
## Correlation with independent Variable 

df2 = df_feature.drop(['user', 'Clicked on Ad', 'Ad Topic Line', 'City'], axis=1)

df2.corrwith(df_feature['Clicked on Ad']).plot.bar(

        figsize = (10, 10), title = "Correlation with Clicked on Ad", fontsize = 15,

        rot = 45, grid = True)
sns.set(style="white")

# Compute the correlation matrix

corr = df2.corr()

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

df_feature.columns

df2 = df_feature.drop(['user', 'Daily Time Spent on Site', 'Daily Internet Usage',

       '% spending time', 'Age', 'Area Income', 'Ad Topic Line', 'City' , 'Country',

       'Timestamp', 'Hour', 'Clicked on Ad'], axis=1)

fig = plt.figure(figsize=(15, 12))

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
df_feature.describe()
df_feature['Clicked on Ad'].value_counts()
countNotClicked = len(df_feature[df_feature['Clicked on Ad'] == 0])     

countClicked  = len(df_feature[df_feature['Clicked on Ad'] == 1]) 

print('Percentage of not Clicked on Ad: {:.2f}%'.format((countNotClicked/len(df_feature)) * 100)) 

print('Percentage of Clicked on Ad: {:.2f}%'.format((countClicked/len(df_feature)) * 100))
df_feature.groupby(df_feature['Clicked on Ad']).mean().head()
sns.heatmap(df_feature.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df_feature.isnull().any()
df_feature.isnull().sum()
null_percentage = (df_feature.isnull().sum()/len(df_feature) * 100)

null_percentage = pd.DataFrame(null_percentage, columns = ['Percentage Null Values (%)'])
null_percentage
df_feature.columns

X = df_feature.drop(['user', 'Clicked on Ad', 'Ad Topic Line', 'City',

              'Country', 'Timestamp'], axis=1)

y = df_feature['Clicked on Ad']
#Get Dummies

X = pd.get_dummies(X)
#Avoiding Dummies Trap

X = X.drop(['continent_not found'], axis=1)

X.isnull().sum()
X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0) 
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)

X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
## Logistic Regression

from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(random_state = 0, penalty = 'l1')

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
## K-Nearest Neighbors (K-NN)

from sklearn.neighbors import KNeighborsClassifier

kn_classifier = KNeighborsClassifier(n_neighbors=35, metric='minkowski', p= 2)

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

svc_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)

svc_linear_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = svc_linear_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## SVM (rbf)

from sklearn.svm import SVC

svc_rbf_classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)

svc_rbf_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = svc_rbf_classifier.predict(X_test)

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
#Installing pydotplus package

!pip install pydotplus
## Plotting Decision Tree

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus



dot_data = StringIO()

export_graphviz(dt_classifier, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())

## Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,

                                    criterion = 'gini')

rf_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = rf_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Gini (n=100)', acc, prec, rec, f1]],

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

                                                  ('svc_linear', svc_linear_classifier),

                                                  ('svc_rbf', svc_rbf_classifier),

                                                  ('gb', gb_classifier),

                                                  ('dt', dt_classifier),

                                                  ('rf', rf_classifier),

                                                  ('ad', ad_classifier),

                                                  ('gr', gr_classifier),

                                                  ('xg', xg_classifier),],

voting='soft')
for clf in (lr_classifier,kn_classifier,svc_linear_classifier,svc_rbf_classifier,

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
results
#The Best Classifier

print('The best classifier is:')

print('{}'.format(results.sort_values(by='Accuracy',ascending=False).head(5)))
#Applying K-fold validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=svc_linear_classifier, X=X_train, y=y_train,cv=10)

accuracies.mean()

accuracies.std()

print("SVM (Linear) Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
## EXTRA: Confusion Matrix

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, annot=True, fmt='g')

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred)) 
#Plotting Cumulative Accuracy Profile (CAP)

y_pred_proba = svc_linear_classifier.predict_proba(X=X_test)

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
#Permutation Importance

import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(svc_linear_classifier, random_state=0).fit(X_test,y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# Analyzing Coefficients

pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]),

           pd.DataFrame(np.transpose(svc_linear_classifier.coef_), columns = ["coef"])

           ],axis = 1)
# Recursive Feature Elimination

from sklearn.feature_selection import RFE

from sklearn.svm import SVC



# Model to Test

classifier = SVC(random_state = 0, kernel = 'linear', probability= True)



# Select Best X Features

rfe = RFE(classifier, n_features_to_select=None)

rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes

print(rfe.support_)

print(rfe.ranking_)

X_train.columns[rfe.support_]
# Fitting Model to the Training Set

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'linear', probability= True)

classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM RFE (Linear)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
# Formatting Final Results

df_feature.columns

user_identifier = df_feature['user']

final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()

final_results['predicted'] = y_pred

final_results = final_results[['user', 'Clicked on Ad', 'predicted']].reset_index(drop=True)
final_results.head()