# Importing essential tools
# Regular EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score,cross_validate,KFold,ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, plot_roc_curve,roc_curve,roc_auc_score
# Feature selection
from sklearn.feature_selection import SelectFromModel
#Pipeline
from sklearn.pipeline import make_pipeline,Pipeline
plt.style.use('seaborn-whitegrid')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../input/heart-disease-uci/heart.csv",sep=",")
df.head()
df.shape #(rows,columns)
df.head()
df.target.value_counts() # balanced dataset
df.target.value_counts().plot.bar();
df.info() # No-Null values and all numerical features
df.isna().sum() # another way of checking missing values
df.describe()
df.sex.value_counts() # 1=male , 0-female
# comparing target `column` with `sex` column
pd.crosstab(df.sex,df.target)
fig, ax = plt.subplots(figsize=(10, 6))

pd.crosstab(df.sex, df.target).plot(kind="bar",
                                    color=["salmon", 'lightblue'],
                                    figsize=(10, 6),
                                    ax=ax);

ax.set(xlabel="Sex (Female-0 , Male-1)",
       ylabel="Heart Disease Frequeny",
       title="Heart disease frequency for sex");

plt.xticks(rotation=0);

ax.legend(['Negative','Positive'],title ="Target");
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(x=df.age,
           y= df.thalach,
           c=df.target,
               cmap='winter');

ax.set(xlabel="Age",ylabel="Max Heaer Rate Achieved",title="Heart Disease in function of Age and Max_Heart_Rate ")
ax.legend(*scatter.legend_elements(),title="Target");
plt.xticks(rotation=0);
df.age.hist(bins= 15); # Helps in checking the ouliers
pd.crosstab(df.ca,df.target,)
fig, ax = plt.subplots(figsize=(10, 6))

pd.crosstab(df.cp,df.target,).plot.bar(color=["salmon","lightblue"],ax=ax)

ax.set(xlabel="Chest Pain type",
       ylabel="Heart Disease Frequeny",
       title="Heart Disease frequency per chest pain type");

plt.xticks(rotation=0);

ax.legend(['Negative','Positive'],title ="Heart Disease");
 
corr_matrix = df.corr()
corr_matrix
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                linewidths=0.5,
                fmt=".2f",
                cmap="YlGnBu")
X,y = df.drop("target",axis=1),df.target
np.random.seed(24)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
models = {
    "Random Forest Classifier    " : RandomForestClassifier(),
    "Logistic Regression" : LogisticRegression(),
    "Knn" : KNeighborsClassifier()
}
score_dict = {}
for i in models:
    models[i].fit(X_train,y_train)
    score_dict[i] = cross_val_score(models[i],X,y).mean();
score_dict
score_df = pd.DataFrame(score_dict,index=['cross_val_score']).T
score_df.plot(kind='bar',figsize=(10,6))
plt.xticks(rotation=0);
tuned_score = {} 

def evaluate(model,a,b):
    metrics=['accuracy','f1','recall','precision','roc_auc']
    
    # cross-validatins the model to calculate scores
    cv = ShuffleSplit(n_splits=5, test_size=0.2,random_state=99)
    score = cross_validate(model,a,b,scoring=metrics,cv= cv)
    
    # rounding off the scores to 2 decimal places
    for i in score:
        score[i] = round(np.array(score[i]).mean(),ndigits=2)
    
    #plotting the cross-validates scores
    score_df = pd.DataFrame(score,index=["score"]).iloc[:,2:].T
    score_df.plot.bar(figsize=(6,5),color=['salmon'],width=0.2)
    plt.xticks(rotation=0) 
    
    return (score) # returning the dictionery of crossvalidation scores for all metrics
# splitting data into features and target
X,y = df.drop("target",axis=1),df.target

# converting the categorical features into dummies and dropping 1 column out of each of them
cat_features = ['cp','ca','slope','thal','sex']
X = pd.get_dummies(X,columns=cat_features)
X = X.drop(['cp_0','ca_0','slope_0','thal_0','sex_0'],axis=1)

# splitting dataset into train and test set
np.random.seed(24)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# scaling the features usng Standard Scaler
scaler = StandardScaler()
X_train_encoded_scaled = scaler.fit_transform(X_train);
X_test_encoded_scaled = scaler.transform(X_test);
# variables for storing train and test score
train_score = np.array([])
test_score = np.array([])

neighbors = np.arange(1,21)
for i in neighbors :
    # initializing model variable with diffrent values of n_neighbors parameter
    knn = KNeighborsClassifier(n_neighbors=i)
    
    # fitting training set data
    knn.fit(X_train_encoded_scaled,y_train)
    
    # appending scores in the respective arrays
    train_score =  np.append(train_score , [knn.score(X_train_encoded_scaled,y_train)])
    test_score =  np.append(test_score , [knn.score(X_test_encoded_scaled,y_test)])
    
# plotting scores 
plt.plot(neighbors,train_score,'b',label="train")
plt.plot(neighbors,test_score,'g',label="test")
plt.legend()
plt.xticks(np.arange(1,21))
plt.title(f'max_score is {test_score.max()}   at  n_neighbor ={np.argmax(test_score)+1}');
# list of parameters and their possible values
parameters = {
    "kneighborsclassifier__n_neighbors" : np.arange(1,15),
    "kneighborsclassifier__leaf_size" : np.arange(1,5),
    "kneighborsclassifier__weights" : ["uniform","distance"]
}

# initializing model variable and implementing GridSearchCV
knn = KNeighborsClassifier()

# creating a pipeline with sacler and estimator 
knn_scaling_pipeline = make_pipeline(StandardScaler(),KNeighborsClassifier())
# implementing GridSearchCV on the pipeline
knn_cv = GridSearchCV(knn_scaling_pipeline,param_grid=parameters,n_jobs=-1,cv=5)

# fitting the training data to the 
knn_cv.fit(X_train,y_train)
best_estimator =  knn_cv.best_estimator_

# creating pipeline with scaler and best_estimator so as to calculate the cross-validation scores in the "evaluate" function
knn_scaling_pipeline = make_pipeline(StandardScaler(), best_estimator)

sc = evaluate(knn_scaling_pipeline,X,y)

# inserting the scores into tuned_scores dict
tuned_score['k-NN'] = sc
sc
# Creating features and label set
X,y = df.drop("target",axis=1),df.target

# splitting dataset into train and test set
np.random.seed(24)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
# list of possible values of the parameters
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90],
    'max_features': [2],
    'min_samples_leaf': [5],
    'min_samples_split': [ 12],
    'n_estimators': np.arange(0,300,50)
}
# initializing model 
rfc =RandomForestClassifier(random_state=42)

# implimenting GridSeachCV and fitting training set to it
rfc_cv = GridSearchCV(rfc,n_jobs=-1,param_grid=param_grid,cv=5)
rfc_cv.fit(X_train,y_train)
best_estimator = rfc_cv.best_estimator_

#evaluating model
sc = evaluate(best_estimator,X,y)

# inserting the scores into tuned_scores dict
tuned_score['RFC'] = sc
sc
rfc_cv.best_estimator_.feature_importances_
# initializing the transformer 
sfm = SelectFromModel(best_estimator,threshold=0.03,prefit=True)
# since estimator is pre-fitted therfore no need to fit it again with the training set data , just make prefit = True 
 
# transforming the train and test dataset , features with importance value above threshold will be selected
X_important = sfm.transform(X)
X_train_important = sfm.transform(X_train)
X_test_important = sfm.transform(X_test)

# fittting the GridSeachCV model with important features
rfc_cv.fit(X_train_important,y_train)

# evaluating model
sc = evaluate(rfc_cv.best_estimator_,X_important,y)

# inserting the scores into tuned_scores dict
tuned_score['RFC_imp_features'] = sc
sc
X,y = df.drop("target",axis=1),df.target
np.random.seed(24)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

scaler = StandardScaler()
X_train__scaled = scaler.fit_transform(X_train);
X_test__scaled = scaler.transform(X_test);
# initializing model and evaluating results on the the scaled data
lr = LogisticRegression()
lr.fit(X_train__scaled,y_train)
print("test-score = ",lr.score(X_test__scaled,y_test),"\n")

# pipeline with scaler and estimator 
lr_scaling_pipeline = make_pipeline(StandardScaler(), lr)
sc = evaluate(lr_scaling_pipeline,X,y)

# inserting the scores into tuned_scores dict
tuned_score['LR-scaled'] = sc
sc
param_grid = {'logisticregression__C': np.logspace(-4,4,30),'logisticregression__solver':['liblinear'],'logisticregression__penalty':['l1','l2']}


# implementing GridSearchCV
lr = LogisticRegression()

# pipeline with scaler and estimator to prevent data leakage while scaling
lr_pipeline = make_pipeline(StandardScaler(), lr)
lr_cv = GridSearchCV(lr_pipeline,cv=5,n_jobs=-1,param_grid=param_grid)

# fitting the training data to lr_cv
lr_cv.fit(X_train,y_train)

# evaluating model using pipleine
lr_cv_pipeline = make_pipeline(StandardScaler(),lr_cv.best_estimator_)
sc = evaluate(lr_cv_pipeline,X,y)

# inserting the scores into tuned_scores dict
tuned_score['LR-Tuned'] =sc
sc
# variable for storing the 2-dimensional array
score_array=np.ndarray((5,5))

# converting the scores in the tuned)score dict into 2-dimensional np.array
for i,j in enumerate(tuned_score):
    score_array[i,:] = list(tuned_score[j].values())[2:]

# initializing scaler object
scaler = MinMaxScaler(feature_range=(0,1))

# MinMax Scaled cross-validation scores
scaled_score_array = np.transpose(scaler.fit_transform(score_array))

# cross-validation scores ( columns - models)(index - metrics)
score_array = np.transpose(score_array)
print(score_array)
# Dataframe with cross-validation scores
score_df = pd.DataFrame(score_array,columns=list(tuned_score.keys()),index= ['accuracy','f1','recall','precision','roc_auc'])

# Dataframe with MinMax scaled cross-validation scores
scaled_score_df = pd.DataFrame(scaled_score_array,columns=list(tuned_score.keys()),index= ['accuracy','f1','recall','precision','roc_auc'])
score_df
scaled_score_df
# Plotting scores
score_df.plot.bar(figsize=(15,4));
plt.yticks(np.linspace(0,1,11,endpoint=True));
plt.xticks(rotation=0,fontsize=12)
plt.xlabel("Metrics",fontsize=15)
plt.ylabel("CV-Score",fontsize=15)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, fancybox=True, shadow=True);
scaled_score_df.plot.bar(figsize=(15,4));
plt.yticks(np.linspace(0,1,11,endpoint=True));
plt.xticks(rotation=0,fontsize=12)
plt.xlabel("Metrics",fontsize=15)
plt.ylabel("Relative CV-Score",fontsize=15)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, fancybox=True, shadow=True);