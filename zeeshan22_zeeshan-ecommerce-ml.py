import numpy as np 
import pandas as pd 
import math

import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import style

style.use('ggplot')



import os 

df=pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
df.info()
df.describe()
cat_cols=[x for x in df.columns if df[x].dtype=='O']
cat_cols
num_cols=[x for x in df.columns if df[x].dtype!='O']
num_cols
for col in df.columns:
    print(col,"\t",df[col].isna().mean()*100)
df['units_sold'].astype('O').value_counts()


## Inspecting the target variable units_sold
plt.figure(figsize=(14,6))
sns.countplot(data=df,x="units_sold")
## 9 significant categories in the units_sold which is a discrete variable the remaining categories can be replaced by the
# values closest to them 1,2,3,6,7,8 are the closest in value to 10 units sold

units_sold_list=list(df['units_sold'].astype('O').value_counts().index)
units_sold_list
for val in units_sold_list:
    if (val == 8 or val ==1 or val==7 or val==3 or val ==2 or val==6):
        df['units_sold']=df['units_sold'].replace(val,10)
       
plt.figure(figsize=(14,6))
sns.countplot(data=df,x="units_sold")
df['product_color'].value_counts()
color_list=list(df['product_color'].value_counts().index)
for x in color_list:
    print(x)
for x in color_list:
    if '&' in x:
        df['product_color']=df['product_color'].replace(x,'dual')
    
for x in color_list:
    if 'green' in x or 'khaki' in x:
         df['product_color']=df['product_color'].replace(x,'green')
for x in color_list:
    if 'blue' in x or 'navy'  in x :
          df['product_color']=df['product_color'].replace(x,'blue')
for x in color_list:
    if 'red' in x or 'rose'  in x :
          df['product_color']=df['product_color'].replace(x,'red')
df['product_color'] = df['product_color'].replace('Black', 'black')
df['product_color'] = df['product_color'].replace('White', 'white')


df['product_color'] = df['product_color'].replace('lightpink', 'pink')

df['product_color'] = df['product_color'].replace('gray', 'grey')

df['product_color'] = df['product_color'].replace('coffee', 'brown')

df['product_color'] = df['product_color'].replace('multicolor', 'other')
df['product_color'] = df['product_color'].replace('floral', 'other')
df['product_color'] = df['product_color'].replace('leopard', 'other')
df['product_color'] = df['product_color'].replace('camouflage', 'other')


    
df['product_color'] =df['product_color'].replace(np.nan, 'other')
df['product_color'] = df['product_color'].replace('apricot', 'other')
df['product_color'] = df['product_color'].replace('camel', 'yellow')
df['product_color'] = df['product_color'].replace('lightyellow', 'yellow')
df['product_color'] = df['product_color'].replace('burgundy', 'red')
df['product_color'] = df['product_color'].replace('wine', 'red')
df['product_color'] = df['product_color'].replace('silver', 'other')
df['product_color'] = df['product_color'].replace('whitefloral', 'white')

df['product_color'] = df['product_color'].replace('RED', 'red')
df['product_color'].value_counts()
df['shipping_is_express'].astype('O').value_counts()
plt.figure(figsize=(14,6))
sns.countplot(x='shipping_is_express',data=df)
##Majority items are not shipped under express shipping
plt.figure(figsize=(14,6))
sns.scatterplot(x='shipping_is_express',y='units_sold',data=df)

## The shipping methods influence on units sold can not be aptly inferred as only four products use express shipping but it 
## can be observed that the highest units sold is for products without express shipping
df['origin_country'].astype('O').value_counts()
plt.figure(figsize=(14,6))
sns.countplot(x='origin_country',data=df)

## Majority of goods are manufactured in China,we can observe China as a manufacturing hub
## US comes in second with 31 products ,the remaining countries can all be clubbed under others as they dont contribute significantly
country_list=list(df['origin_country'].astype('O').value_counts().index)
country_list
for x in country_list:
    if x!='US'and x!='CN':
        df['origin_country']=df['origin_country'].replace(x,'OTHER')
        
df['origin_country']=df['origin_country'].replace(np.nan,'OTHER')    
df['origin_country'].astype('O').value_counts()
plt.figure(figsize=(14,6))
sns.countplot(x='origin_country',data=df)
def null_val_viz(df):
    cols_with_null=[col for col in df.columns if df[col].isna().mean()>0]
    pct_null_vals=df[cols_with_null].isna().mean()*100 
    plt.figure(figsize=(14,6))
    sns.barplot(y=pct_null_vals.index,x=pct_null_vals,orient='h')
    plt.title("NA VALUES BY COLUMN")
    
null_val_viz(df)

plt.figure(figsize=(14,6))
sns.scatterplot(data=df,x="price",y="units_sold")

## products between 1 and 10 euros have a few products with 100000 units sold and products priced between 10 and 20 euros cap
## at about 50000 units sold
### products priced between 20 and 30 euros cap at 20000 units sold while premium prodcuts priced slightly less than 50 euros 
## has one unit sold
## Highest variability in sales is between low and mid priced products
df["units_sold"].describe()
plt.figure(figsize=(14,6))
sns.scatterplot(data=df,x="rating",y="units_sold")
plt.figure(figsize=(14,6))
sns.distplot(df['price'], color='green', label='Price')
sns.distplot(df['retail_price'], color='yellow', label='Retail price')
plt.legend();

 
plt.figure(figsize=(14,6))
sns.distplot(df['units_sold'].astype('O'))

pd.crosstab(df['units_sold'],df['uses_ad_boosts'])
df_use_boost=df[df['uses_ad_boosts']== 1]
df_not_use_boost=df[df['uses_ad_boosts']== 0]

from scipy.stats import ttest_ind
ttest_ind(df_use_boost['units_sold'],df_not_use_boost['units_sold'])
## H0:Mean units sold for products with ad boosts - Mean units sold for products without ad boosts == 0
##HA:Mean units sold for products with ad boosts - Mean units sold for products without ad boosts  !=0
###Difference in mean units sold for populations using ad boosts and  not using ad boosts
## A very high p value suggests we cannot reject the null hypothesis aka zero differences in 
## mean units_sold by presence and absence of ad boosts
df['badge_local_product'].astype('O').value_counts()
local_vec=df['badge_local_product'].astype('O')
plt.figure(figsize=(14,6))
sns.scatterplot(x=local_vec,y=df["units_sold"])
df_local_product_badge=df[df['badge_local_product']== 1]
df_no_local_product_badge=df[df['badge_local_product']== 0]
ttest_ind(df_local_product_badge['units_sold'],df_no_local_product_badge['units_sold'])
## H0:Mean units sold for productis with local badge - Mean units sold for productis without local badge == 0
##HA:Mean units sold for productis with local badge - Mean units sold for productis without local badge != 0
## p value greater than 0.05 we cannot reject the null hyptothesis hence there's no statistically significant effect of the 
## presence of local product badge on units sold
df['badge_product_quality'].astype('O').value_counts()
df_has_quality_badge=df[df['badge_product_quality']== 1]
df_no_quality_badge=df[df['badge_product_quality']== 0]
ttest_ind(df_has_quality_badge['units_sold'],df_no_quality_badge['units_sold'])
## H0:Mean units sold for productis with quality badge - Mean units sold for productis without quality badge == 0
##HA:Mean units sold for productis with quality badge - Mean units sold for productis without quality badge != 0
## p value less than 0.05 we can reject the null hyptothesis hence there's a statistically significant effect of the 
## presence of quality badge on units sold
plt.figure(figsize=(14,6))
sns.scatterplot(x=df['badge_product_quality'].astype('O'),y=df["units_sold"])
df['product_variation_size_id'].value_counts()
df['product_variation_size_id'].value_counts().index
df['product_variation_size_id'] = df['product_variation_size_id'].replace('Size -XXS', 'S')
df['product_variation_size_id'] = df['product_variation_size_id'].replace('SIZE-XXS', 'S')
df['product_variation_size_id'] = df['product_variation_size_id'].replace('Size S.', 'S')
df['product_variation_size_id'] = df['product_variation_size_id'].replace('s', 'S')
df['product_variation_size_id'] = df['product_variation_size_id'].replace('SizeL', 'L')

df['product_variation_size_id'] =df['product_variation_size_id'].replace('5XL', 'XL')
df['product_variation_size_id'] = df['product_variation_size_id'].replace('4XL', 'XL')
df['product_variation_size_id'] = df['product_variation_size_id'].replace('3XL', 'XL')
df['product_variation_size_id'] = df['product_variation_size_id'].replace('2XL', 'XL')
for x in df['product_variation_size_id'].value_counts().index:
     if x != 'XXXS' \
    and x != 'XXS' \
    and x != 'XS' \
    and x != 'S' \
    and x != 'M' \
    and x != 'L' \
    and x != 'XL' \
    and x != 'XXL' \
    and x != 'XXXXL' \
    and x != 'XXXXXL':
            df['product_variation_size_id']= df['product_variation_size_id'].replace(x,'other')
           
df['product_variation_size_id']= df['product_variation_size_id'].replace(np.nan,'other')
df['product_variation_size_id'].value_counts()
plt.figure(figsize=(14,6))
sns.countplot(data=df,x='product_variation_size_id',order=df['product_variation_size_id'].value_counts().index)
fig,ax=plt.subplots(figsize=(27,27))
fig=sns.heatmap(data=df.corr(),annot=True,cbar=True,vmin=0,vmax=1,ax=ax)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.show()
df['has_urgency_banner'].replace(np.nan,0,inplace=True)
df.drop(['title','title_orig','currency_buyer','merchant_title','crawl_month','product_id','product_picture', 
         'product_url', 'merchant_profile_picture', 'merchant_id', 'currency_buyer','theme','urgency_text',
         'merchant_info_subtitle','title','title_orig','tags','shipping_option_name'],axis=1,inplace=True)
df.drop(['rating_count'],axis=1,inplace=True)
## Rating count suggests the number of ratings for every product but the same information is conveyed by the
## breakdown of the number of each type of rating rating_five_count to rating_one_count columns,hence due to redundancy
## we can drop this column.
df.drop(['merchant_name'],axis=1,inplace=True)
df['rating_five_count'].describe()
df.isna().sum()
df['rating_four_count'].replace(np.nan,0,inplace=True)
df['rating_five_count'].replace(np.nan,0,inplace=True)
df['rating_three_count'].replace(np.nan,0,inplace=True)
df['rating_two_count'].replace(np.nan,0,inplace=True)
df['rating_one_count'].replace(np.nan,0,inplace=True)
df.isna().sum()
df=pd.get_dummies(df,columns=['product_color'],prefix='COLOR_',drop_first=True)


df=pd.get_dummies(df,columns=['origin_country'],prefix='origin_',drop_first=True)
df.columns
df=pd.get_dummies(df,columns = ['product_variation_size_id'],prefix = 'prodsize_',drop_first = True)
df.head(20)
df.to_csv('preprocessed_ecom.csv')
X=df.drop(['units_sold'],axis=1)
y=df['units_sold']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import GridSearchCV
lr=lr(max_iter=10000,multi_class='multinomial',solver='saga',random_state = 10)
lr.fit(X_train,y_train)



print(f"Training data accuracy : {lr.score(X_train,y_train)}")

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred_lr = lr.predict(X_test)
print(f"Test data accuracy : {accuracy_score(y_test, y_pred_lr)}")

from sklearn.tree import DecisionTreeClassifier

classifier_DT = DecisionTreeClassifier(random_state = 10)
classifier_DT.fit(X_train, y_train)

from sklearn import tree
tree.plot_tree(classifier_DT)
from sklearn.model_selection import cross_val_score
print(f"Training data cross val accuracy :{np.mean(cross_val_score(classifier_DT,X_train,y_train,cv=5))}")

y_pred_DT = classifier_DT.predict(X_test)
print(f"Test data accuracy : {accuracy_score(y_test, y_pred_DT)}")
from xgboost import XGBClassifier
XGB=XGBClassifier()
XGB.fit(X_train,y_train)
print(f"Training data cross val accuracy :{np.mean(cross_val_score(XGB,X_train,y_train,cv=5))}")
##XGB tends to overfit slightly
y_pred_DT = classifier_DT.predict(X_test)
print(f"Test data accuracy : {accuracy_score(y_test, y_pred_DT)}")
from sklearn.ensemble import AdaBoostClassifier
ADB_Classifier=AdaBoostClassifier()
ADB_Classifier.fit(X_train,y_train)
print(f"Training data cross val accuracy :{np.mean(cross_val_score(ADB_Classifier,X_train,y_train,cv=5))}")
y_pred_ADB = ADB_Classifier.predict(X_test)
print(f"Test data accuracy : {accuracy_score(y_test, y_pred_ADB)}")
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)

print(f"Training data cross val accuracy :{np.mean(cross_val_score(RFC,X_train,y_train,cv=5))}")
y_pred_RFC = RFC.predict(X_test)
print(f"Test data accuracy : {accuracy_score(y_test, y_pred_RFC)}")
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()
GBC.fit(X_train, y_train)
print(f"Training data cross val accuracy :{np.mean(cross_val_score(GBC,X_train,y_train,cv=5))}")
y_pred_GBC = GBC.predict(X_test)
print(f"Test data accuracy : {accuracy_score(y_test, y_pred_GBC)}")
classifiers=[classifier_DT,GBC,RFC,ADB_Classifier,XGB]
name_keys=["Decision_Tree","Gradient_Boosting_Classifier","Random_Forest_Classifier","AdaBoostClassifier","XGBoost"]
mean_cross_val_score=[]
for clf in classifiers:
    accuracies = cross_val_score(estimator = clf, 
                                 X = X_train, 
                                 y = y_train, 
                                 cv = 5)
    mean_cross_val_score.append(np.mean(accuracies)*100)

    
accuracies = pd.DataFrame({'Classifier': name_keys,
                           'mean_cross_val_score': mean_cross_val_score})
accuracies.sort_values('mean_cross_val_score',ascending=False)
    
from sklearn.model_selection import GridSearchCV
param_grid_RFC=[{"criterion":["gini","entropy"],"n_estimators":[100,200,300,500],"max_features":["auto","sqrt","log2"]}]
RFC_grid=GridSearchCV(estimator=RFC,param_grid=param_grid_RFC,cv=5,return_train_score=True)
RFC_grid.fit(X_train,y_train)
print(f"The optimum hyperparameters to be used : {RFC_grid.best_params_}")
print(f"The best cross val score :{RFC_grid.best_score_}")
y_RFC_tuned=RFC_grid.predict(X_test)

print(f"Test data accuracy : {accuracy_score(y_test, y_RFC_tuned)}")
