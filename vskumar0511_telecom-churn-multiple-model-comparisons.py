# Importing Pandas and NumPy

import pandas as pd,numpy as np



# Plot related packages

import matplotlib.pyplot as plt,seaborn as sns; sns.set()



# Clustering packages (for Predicting a label for the unsupervised data)

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering



# Model Building related packages for Classification (Supervised)

from sklearn.ensemble import  RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB,BernoulliNB

from sklearn.model_selection import train_test_split



from sklearn import preprocessing #For scaling



from sklearn.preprocessing import LabelEncoder # Converting Categorical(text) to Categorical(numerical)



# PCA package

from sklearn.decomposition import PCA



# Metrics



from sklearn import metrics



# import package to avoid warnings 

import warnings

warnings.filterwarnings("ignore")
# Importing all datasets

churn_data = pd.read_csv("../input/churn_data.csv")

customer_data = pd.read_csv("../input/customer_data.csv")

internet_data = pd.read_csv("../input/internet_data.csv")
churn_data.shape
#Merging on 'customerID'

df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')
#Final dataframe with all predictor variables

telecom = pd.merge(df_1, internet_data, how='inner', on='customerID')
# Let's see the head of our master dataset

telecom.head()
telecom.describe()
# Let's see the type of each column

telecom.info()
# Converting Yes to 1 and No to 0

telecom['PhoneService'] = telecom['PhoneService'].map({'Yes': 1, 'No': 0})

telecom['PaperlessBilling'] = telecom['PaperlessBilling'].map({'Yes': 1, 'No': 0})

telecom['Churn'] = telecom['Churn'].map({'Yes': 1, 'No': 0})

telecom['Partner'] = telecom['Partner'].map({'Yes': 1, 'No': 0})

telecom['Dependents'] = telecom['Dependents'].map({'Yes': 1, 'No': 0})
# Creating a dummy variable for the variable 'Contract' and dropping the first one.

cont = pd.get_dummies(telecom['Contract'],prefix='Contract',drop_first=True)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,cont],axis=1)



# Creating a dummy variable for the variable 'PaymentMethod' and dropping the first one.

pm = pd.get_dummies(telecom['PaymentMethod'],prefix='PaymentMethod',drop_first=True)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,pm],axis=1)



# Creating a dummy variable for the variable 'gender' and dropping the first one.

gen = pd.get_dummies(telecom['gender'],prefix='gender',drop_first=True)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,gen],axis=1)



# Creating a dummy variable for the variable 'MultipleLines' and dropping the first one.

ml = pd.get_dummies(telecom['MultipleLines'],prefix='MultipleLines')

#  dropping MultipleLines_No phone service column

ml1 = ml.drop(['MultipleLines_No phone service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,ml1],axis=1)



# Creating a dummy variable for the variable 'InternetService' and dropping the first one.

iser = pd.get_dummies(telecom['InternetService'],prefix='InternetService',drop_first=True)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,iser],axis=1)



# Creating a dummy variable for the variable 'OnlineSecurity'.

os = pd.get_dummies(telecom['OnlineSecurity'],prefix='OnlineSecurity')

os1= os.drop(['OnlineSecurity_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,os1],axis=1)



# Creating a dummy variable for the variable 'OnlineBackup'.

ob =pd.get_dummies(telecom['OnlineBackup'],prefix='OnlineBackup')

ob1 =ob.drop(['OnlineBackup_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,ob1],axis=1)



# Creating a dummy variable for the variable 'DeviceProtection'. 

dp =pd.get_dummies(telecom['DeviceProtection'],prefix='DeviceProtection')

dp1 = dp.drop(['DeviceProtection_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,dp1],axis=1)



# Creating a dummy variable for the variable 'TechSupport'. 

ts =pd.get_dummies(telecom['TechSupport'],prefix='TechSupport')

ts1 = ts.drop(['TechSupport_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,ts1],axis=1)



# Creating a dummy variable for the variable 'StreamingTV'.

st =pd.get_dummies(telecom['StreamingTV'],prefix='StreamingTV')

st1 = st.drop(['StreamingTV_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,st1],axis=1)



# Creating a dummy variable for the variable 'StreamingMovies'. 

sm =pd.get_dummies(telecom['StreamingMovies'],prefix='StreamingMovies')

sm1 = sm.drop(['StreamingMovies_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,sm1],axis=1)
#telecom['MultipleLines'].value_counts()
# We have created dummies for the below variables, so we can drop them

telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)
#The varaible was imported as a string we need to convert it to float

telecom['TotalCharges'] =telecom['TotalCharges'].convert_objects(convert_numeric=True)

#telecom['tenure'] = telecom['tenure'].astype(int).astype(float)
telecom.info()
# Checking for outliers in the continuous variables

num_telecom = telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]
# Checking outliers at 25%,50%,75%,90%,95% and 99%

num_telecom.describe(percentiles=[.25,.5,.75,.90,.95,.99])
# Adding up the missing values (column-wise)

telecom.isnull().sum()
# Checking the percentage of missing values

round(100*(telecom.isnull().sum()/len(telecom.index)), 2)
a=telecom.isnull().sum()

a[a>0]
# Removing NaN TotalCharges rows

telecom = telecom[~np.isnan(telecom['TotalCharges'])]
a=telecom.isnull().sum()

a[a>0]
telecom.reset_index(inplace=True)
# Checking percentage of missing values after removing the missing values

round(100*(telecom.isnull().sum()/len(telecom.index)), 2)
telecom=telecom.drop(["customerID","index"],1)
telecom.shape
# Normalising continuous features

df = telecom[['tenure','MonthlyCharges','TotalCharges']]
df.shape
from sklearn.preprocessing import StandardScaler



normalized_df=pd.DataFrame(StandardScaler().fit_transform(df),columns=df.columns)
normalized_df.index
telecom = telecom.drop(['tenure','MonthlyCharges','TotalCharges'], 1)
telecom=pd.concat([telecom,normalized_df],axis=1)
telecom.shape
telecom1=telecom.drop(["Churn"],1)
X = telecom1.copy()

Y = telecom["Churn"]
X.head()
# Kmeans iteration 1 till 20 cluster size



cluster_errors = []



for i in range(1,21):

    clusters = KMeans(i)

    clusters.fit(X)

    cluster_errors.append(clusters.inertia_)

    



# WSS values



clusters_df = pd.DataFrame({"Num_clusters":range(1,21),"cluster_errors":cluster_errors})



# WSS values vs Number of Clusters (Elbow graph)



plt.figure(figsize=(10,5))

sns.pointplot(x=clusters_df.Num_clusters,y=clusters_df.cluster_errors,data=clusters_df)

plt.xticks(range(20))

plt.show()
# Fit the data and check the accuracy score



kmean = KMeans(n_clusters=2).fit(X)



y_kmean = list(kmean.labels_)



kmean_met = metrics.accuracy_score(Y,y_kmean)

kmean_met
cluster = AgglomerativeClustering(n_clusters = 2, linkage="ward")



y_hier = list(cluster.fit_predict(X))



hier_met = metrics.accuracy_score(Y,y_hier)

hier_met 
compr = pd.DataFrame({"Accuracy_Scores":[kmean_met,hier_met]},index=["kmean_cluster","Agglomerative_cluster"])

compr.sort_values(by="Accuracy_Scores",ascending=False)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y_kmean,test_size=0.3, random_state=0)  
#PCA for all the features



pca = PCA()  

X_train = pca.fit_transform(X_train)  

X_test = pca.transform(X_test)  
# Loading Score

print("Loading Scores are:\n",pca.explained_variance_ratio_ )
#Explained variance vs Number of components plot



plt.figure(figsize=(10,3))

plt.plot(pca.explained_variance_ratio_,marker="o",c="r")

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.xticks(range(30))

plt.show()
# Splitting the dataset into the Training set and Test set and applying PCA

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y_kmean,test_size=0.3, random_state=0)  

pca = PCA(n_components=2)  

X_train = pca.fit_transform(X_train)  

X_test = pca.transform(X_test)  
# Random Forest model

random_us = RandomForestClassifier()  

random_us.fit(X_train, y_train)



y_random_unscaled = random_us.predict(X_test) 

y_random_met=metrics.accuracy_score(y_test,y_random_unscaled)

y_random_met_training=random_us.score(X_train, y_train)



print("Test Accuracy for y_random_met:\n",y_random_met)

print()

print("Training Accuracy for y_random_met:\n",y_random_met_training)
# Logistic Regression model



log_model_us = LogisticRegression()  

log_model_us.fit(X_train, y_train)



log_model_unscaled = log_model_us.predict(X_test) 

y_log_model_met=metrics.accuracy_score(y_test,log_model_unscaled)

y_log_model_met_training=log_model_us.score(X_train, y_train)





print("Test Accuracy for y_log_model_met:\n",y_log_model_met)

print()

print("Training Accuracy for y_log_model_met:\n",y_log_model_met_training)
# Decision Tree model # Gini



tree_model_us_gini = DecisionTreeClassifier()  

tree_model_us_gini.fit(X_train, y_train)



tree_model_scaled_gini = tree_model_us_gini.predict(X_test) 

tree_model_met_gini=metrics.accuracy_score(y_test,tree_model_scaled_gini)

tree_model_met_gini_training=tree_model_us_gini.score(X_train, y_train)





print("Test Accuracy for tree_model_Gini_met:\n",tree_model_met_gini)

print()

print("Training Accuracy for tree_model_Gini_met:\n",tree_model_met_gini_training)
# Decision Tree model # Entropy



tree_model_us_entropy = DecisionTreeClassifier(criterion="entropy")  

tree_model_us_entropy.fit(X_train, y_train)



tree_model_scaled_entropy = tree_model_us_entropy.predict(X_test) 

tree_model_met_entropy=metrics.accuracy_score(y_test,tree_model_scaled_entropy)

tree_model_met_entropy_training=tree_model_us_entropy.score(X_train, y_train)



print("Test Accuracy for tree_model_Entropy_met:\n",tree_model_met_entropy)

print()

print("Training Accuracy for tree_model_Entropy_met:\n",tree_model_met_entropy_training)
# KNN model



KNN_model_us = KNeighborsClassifier()  

KNN_model_us.fit(X_train, y_train)



KNN_model_scaled = KNN_model_us.predict(X_test) 

KNN_model_met=metrics.accuracy_score(y_test,KNN_model_scaled)

KNN_model_met_training=KNN_model_us.score(X_train, y_train)





print("Test Accuracy for KNN_model_met:\n",KNN_model_met)

print()

print("Training Accuracy for KNN_model_met:\n",KNN_model_met_training)
# Naive model



Naive_model_us = GaussianNB()  

Naive_model_us.fit(X_train, y_train)



Naive_model_scaled = Naive_model_us.predict(X_test) 

Naive_model_met=metrics.accuracy_score(y_test,Naive_model_scaled)

Naive_model_met_training=Naive_model_us.score(X_train, y_train)



print("Test Accuracy for Naive_model_met:\n",Naive_model_met)

print()

print("Training Accuracy for Naive_model_met:\n",Naive_model_met_training)
# Naive model



Naive_model_us_ber = BernoulliNB()  

Naive_model_us_ber.fit(X_train, y_train)



Naive_model_unscaled_ber = Naive_model_us_ber.predict(X_test) 

Naive_model_met_ber=metrics.accuracy_score(y_test,Naive_model_unscaled_ber)

Naive_model_met_ber_tr=Naive_model_us_ber.score(X_train, y_train)





print("Test Accuracy for Naive_model_met_bernouli:\n",Naive_model_met_ber)

print()

print("Training Accuracy for Naive_model_met_bernouli:\n",Naive_model_met_ber_tr)
# SVM model



SVM_model_us = SVC()  

SVM_model_us.fit(X_train, y_train)



SVM_model_unscaled = SVM_model_us.predict(X_test) 

SVM_model_met=metrics.accuracy_score(y_test,SVM_model_unscaled)

SVM_model_met_tr=SVM_model_us.score(X_train, y_train)





print("Test Accuracy for SVM_model_met:\n",SVM_model_met)

print()

print("Training Accuracy for SVM_model_met:\n",SVM_model_met_tr)
Accuracy_Scores_2=pd.DataFrame([y_random_met,y_log_model_met,tree_model_met_gini,tree_model_met_entropy,

                                     KNN_model_met,Naive_model_met,Naive_model_met_ber,SVM_model_met],columns=["Test_Accuracy_USL_PCA"]

                               ,index= ["Random_met","Log_model_met","tree_model_met_gini","tree_model_met_entropy",

                                     "KNN_model_met","Naive_model_met","Naive_model_met_ber","SVM_model_met"])



Accuracy_Scores_2.sort_values(by="Test_Accuracy_USL_PCA",ascending=False)
Accuracy_Scores_2["Training_Accuracy_USL_PCA"] = [y_random_met_training,y_log_model_met_training,

                                          tree_model_met_gini_training,tree_model_met_entropy_training,

                                     KNN_model_met_training,Naive_model_met_training,

                                          Naive_model_met_ber_tr,SVM_model_met_tr]
Accuracy_Scores_2.sort_values(by="Training_Accuracy_USL_PCA",ascending=False)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y_kmean,test_size=0.3, random_state=0)  
# Random Forest model

random_us1 = RandomForestClassifier()  

random_us1.fit(X_train, y_train)



y_random_unscaled1 = random_us1.predict(X_test) 

y_random_met1=metrics.accuracy_score(y_test,y_random_unscaled1)

y_random_met_training1=random_us1.score(X_train, y_train)



print("Test Accuracy for y_random_met:\n",y_random_met1)

print()

print("Training Accuracy for y_random_met:\n",y_random_met_training1)
# Logistic Regression model



log_model_us1 = LogisticRegression()  

log_model_us1.fit(X_train, y_train)



log_model_unscaled1 = log_model_us1.predict(X_test) 

y_log_model_met1=metrics.accuracy_score(y_test,log_model_unscaled1)

y_log_model_met_training1=log_model_us1.score(X_train, y_train)





print("Test Accuracy for y_log_model_met:\n",y_log_model_met1)

print()

print("Training Accuracy for y_log_model_met:\n",y_log_model_met_training1)
# Decision Tree model # Gini



tree_model_us_gini1 = DecisionTreeClassifier()  

tree_model_us_gini1.fit(X_train, y_train)



tree_model_scaled_gini1 = tree_model_us_gini1.predict(X_test) 

tree_model_met_gini1=metrics.accuracy_score(y_test,tree_model_scaled_gini1)

tree_model_met_gini_training1=tree_model_us_gini1.score(X_train, y_train)





print("Test Accuracy for tree_model_Gini_met:\n",tree_model_met_gini1)

print()

print("Training Accuracy for tree_model_Gini_met:\n",tree_model_met_gini_training1)
# Decision Tree model # Entropy



tree_model_us_entropy1 = DecisionTreeClassifier(criterion="entropy")  

tree_model_us_entropy1.fit(X_train, y_train)



tree_model_scaled_entropy1 = tree_model_us_entropy1.predict(X_test) 

tree_model_met_entropy1=metrics.accuracy_score(y_test,tree_model_scaled_entropy1)

tree_model_met_entropy_training1=tree_model_us_entropy1.score(X_train, y_train)



print("Test Accuracy for tree_model_Entropy_met:\n",tree_model_met_entropy1)

print()

print("Training Accuracy for tree_model_Entropy_met:\n",tree_model_met_entropy_training1)
# KNN model



KNN_model_us1 = KNeighborsClassifier()  

KNN_model_us1.fit(X_train, y_train)



KNN_model_scaled1 = KNN_model_us1.predict(X_test) 

KNN_model_met1=metrics.accuracy_score(y_test,KNN_model_scaled1)

KNN_model_met_training1=KNN_model_us1.score(X_train, y_train)





print("Test Accuracy for KNN_model_met:\n",KNN_model_met1)

print()

print("Training Accuracy for KNN_model_met:\n",KNN_model_met_training1)
# Naive model



Naive_model_us1 = GaussianNB()  

Naive_model_us1.fit(X_train, y_train)



Naive_model_scaled1 = Naive_model_us1.predict(X_test) 

Naive_model_met1=metrics.accuracy_score(y_test,Naive_model_scaled1)

Naive_model_met_training1=Naive_model_us1.score(X_train, y_train)





print("Test Accuracy for Naive_model_met:\n",Naive_model_met1)

print()

print("Training Accuracy for Naive_model_met:\n",Naive_model_met_training1)
# Naive model



Naive_model_us_ber1 = BernoulliNB()  

Naive_model_us_ber1.fit(X_train, y_train)



Naive_model_unscaled_ber1 = Naive_model_us_ber1.predict(X_test) 

Naive_model_met_ber1=metrics.accuracy_score(y_test,Naive_model_unscaled_ber1)

Naive_model_met_ber_tr1=Naive_model_us_ber1.score(X_train, y_train)





print("Test Accuracy for Naive_model_met_bernouli:\n",Naive_model_met_ber1)

print()

print("Training Accuracy for Naive_model_met_bernouli:\n",Naive_model_met_ber_tr1)
# SVM model



SVM_model_us1 = SVC()  

SVM_model_us1.fit(X_train, y_train)



SVM_model_unscaled1 = SVM_model_us1.predict(X_test) 

SVM_model_met1=metrics.accuracy_score(y_test,SVM_model_unscaled1)

SVM_model_met_tr1=SVM_model_us1.score(X_train, y_train)





print("Test Accuracy for SVM_model_met:\n",SVM_model_met1)

print()

print("Training Accuracy for SVM_model_met:\n",SVM_model_met_tr1)
Accuracy_Scores_3=pd.DataFrame([y_random_met1,y_log_model_met1,tree_model_met_gini1,tree_model_met_entropy1,

                                     KNN_model_met1,Naive_model_met1,Naive_model_met_ber1,SVM_model_met1],columns=["Test_Accuracy_USL_No_PCA"]

                               ,index= ["Random_met","Log_model_met","tree_model_met_gini","tree_model_met_entropy",

                                     "KNN_model_met","Naive_model_met","Naive_model_met_ber","SVM_model_met"])



Accuracy_Scores_3.sort_values(by="Test_Accuracy_USL_No_PCA",ascending=False)
Accuracy_Scores_3["Training_Accuracy_USL_No_PCA"] = [y_random_met_training1,y_log_model_met_training1,

                                          tree_model_met_gini_training1,tree_model_met_entropy_training1,

                                     KNN_model_met_training1,Naive_model_met_training1,

                                          Naive_model_met_ber_tr1,SVM_model_met_tr1]
Accuracy_Scores_3.sort_values(by="Training_Accuracy_USL_No_PCA",ascending=False)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)  
# Random Forest model

random_us2 = RandomForestClassifier()  

random_us2.fit(X_train, y_train)



y_random_unscaled2 = random_us2.predict(X_test) 

y_random_met2=metrics.accuracy_score(y_test,y_random_unscaled2)

y_random_met_training2=random_us2.score(X_train, y_train)



print("Test Accuracy for y_random_met:\n",y_random_met2)

print()

print("Training Accuracy for y_random_met:\n",y_random_met_training2)
# Logistic Regression model



log_model_us2 = LogisticRegression()  

log_model_us2.fit(X_train, y_train)



log_model_unscaled2 = log_model_us2.predict(X_test) 

y_log_model_met2=metrics.accuracy_score(y_test,log_model_unscaled2)

y_log_model_met_training2=log_model_us2.score(X_train, y_train)





print("Test Accuracy for y_log_model_met:\n",y_log_model_met2)

print()

print("Training Accuracy for y_log_model_met:\n",y_log_model_met_training2)
# Decision Tree model # Gini



tree_model_us_gini2 = DecisionTreeClassifier()  

tree_model_us_gini2.fit(X_train, y_train)



tree_model_scaled_gini2 = tree_model_us_gini2.predict(X_test) 

tree_model_met_gini2=metrics.accuracy_score(y_test,tree_model_scaled_gini2)

tree_model_met_gini_training2=tree_model_us_gini2.score(X_train, y_train)





print("Test Accuracy for tree_model_Gini_met:\n",tree_model_met_gini2)

print()

print("Training Accuracy for tree_model_Gini_met:\n",tree_model_met_gini_training2)
# Decision Tree model # Entropy



tree_model_us_entropy2 = DecisionTreeClassifier(criterion="entropy")  

tree_model_us_entropy2.fit(X_train, y_train)



tree_model_scaled_entropy2 = tree_model_us_entropy2.predict(X_test) 

tree_model_met_entropy2=metrics.accuracy_score(y_test,tree_model_scaled_entropy2)

tree_model_met_entropy_training2=tree_model_us_entropy2.score(X_train, y_train)



print("Test Accuracy for tree_model_Entropy_met:\n",tree_model_met_entropy2)

print()

print("Training Accuracy for tree_model_Entropy_met:\n",tree_model_met_entropy_training2)
# KNN model



KNN_model_us2 = KNeighborsClassifier()  

KNN_model_us2.fit(X_train, y_train)



KNN_model_scaled2 = KNN_model_us2.predict(X_test) 

KNN_model_met2=metrics.accuracy_score(y_test,KNN_model_scaled2)

KNN_model_met_training2=KNN_model_us2.score(X_train, y_train)





print("Test Accuracy for KNN_model_met:\n",KNN_model_met2)

print()

print("Training Accuracy for KNN_model_met:\n",KNN_model_met_training2)
# Naive model



Naive_model_us2 = GaussianNB()  

Naive_model_us2.fit(X_train, y_train)



Naive_model_scaled2 = Naive_model_us2.predict(X_test) 

Naive_model_met2=metrics.accuracy_score(y_test,Naive_model_scaled2)

Naive_model_met_training2=Naive_model_us2.score(X_train, y_train)





print("Test Accuracy for Naive_model_met:\n",Naive_model_met2)

print()

print("Training Accuracy for Naive_model_met:\n",Naive_model_met_training2)
# Naive model



Naive_model_us_ber2 = BernoulliNB()  

Naive_model_us_ber2.fit(X_train, y_train)



Naive_model_unscaled_ber2 = Naive_model_us_ber2.predict(X_test) 

Naive_model_met_ber2=metrics.accuracy_score(y_test,Naive_model_unscaled_ber)

Naive_model_met_ber_tr2=Naive_model_us_ber2.score(X_train, y_train)





print("Test Accuracy for Naive_model_met_bernouli:\n",Naive_model_met_ber2)

print()

print("Training Accuracy for Naive_model_met_bernouli:\n",Naive_model_met_ber_tr2)
# SVM model



SVM_model_us2 = SVC()  

SVM_model_us2.fit(X_train, y_train)



SVM_model_unscaled2 = SVM_model_us2.predict(X_test) 

SVM_model_met2=metrics.accuracy_score(y_test,SVM_model_unscaled2)

SVM_model_met_tr2=SVM_model_us2.score(X_train, y_train)





print("Test Accuracy for SVM_model_met:\n",SVM_model_met2)

print()

print("Training Accuracy for SVM_model_met:\n",SVM_model_met_tr2)
Accuracy_Scores_4=pd.DataFrame([y_random_met2,y_log_model_met2,tree_model_met_gini2,tree_model_met_entropy2,

                                     KNN_model_met2,Naive_model_met2,Naive_model_met_ber2,SVM_model_met2],columns=["Test_Accuracy_SL_No_PCA"]

                               ,index= ["Random_met","Log_model_met","tree_model_met_gini","tree_model_met_entropy",

                                     "KNN_model_met","Naive_model_met","Naive_model_met_ber","SVM_model_met"])



Accuracy_Scores_4.sort_values(by="Test_Accuracy_SL_No_PCA",ascending=False)
Accuracy_Scores_4["Training_Accuracy_SL_No_PCA"] = [y_random_met_training2,y_log_model_met_training2,

                                          tree_model_met_gini_training2,tree_model_met_entropy_training2,

                                     KNN_model_met_training2,Naive_model_met_training2,

                                          Naive_model_met_ber_tr2,SVM_model_met_tr2]

Accuracy_Scores_4.sort_values(by="Test_Accuracy_SL_No_PCA",ascending=False)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)  
#PCA for all the features



pca = PCA()  

X_train = pca.fit_transform(X_train)  

X_test = pca.transform(X_test)  
# Loading Score

print("Loading Scores are:\n",pca.explained_variance_ratio_ )
#Explained variance vs Number of components plot



plt.figure(figsize=(10,3))

plt.plot(pca.explained_variance_ratio_,marker="o",c="r")

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.xticks(range(30))

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)



pca = PCA(n_components=2)  

X_train = pca.fit_transform(X_train)  

X_test = pca.transform(X_test)
# Random Forest model

random_us3 = RandomForestClassifier()  

random_us3.fit(X_train, y_train)



y_random_unscaled3 = random_us3.predict(X_test) 

y_random_met3=metrics.accuracy_score(y_test,y_random_unscaled3)

y_random_met_training3=random_us3.score(X_train, y_train)



print("Test Accuracy for y_random_met:\n",y_random_met3)

print()

print("Training Accuracy for y_random_met:\n",y_random_met_training3)



# Logistic Regression model



log_model_us3 = LogisticRegression()  

log_model_us3.fit(X_train, y_train)



log_model_unscaled3 = log_model_us3.predict(X_test) 

y_log_model_met3=metrics.accuracy_score(y_test,log_model_unscaled3)

y_log_model_met_training3=log_model_us3.score(X_train, y_train)





print("Test Accuracy for y_log_model_met:\n",y_log_model_met3)

print()

print("Training Accuracy for y_log_model_met:\n",y_log_model_met_training3)



# Decision Tree model # Gini



tree_model_us_gini3 = DecisionTreeClassifier()  

tree_model_us_gini3.fit(X_train, y_train)



tree_model_scaled_gini3 = tree_model_us_gini3.predict(X_test) 

tree_model_met_gini3=metrics.accuracy_score(y_test,tree_model_scaled_gini3)

tree_model_met_gini_training3=tree_model_us_gini3.score(X_train, y_train)





print("Test Accuracy for tree_model_Gini_met:\n",tree_model_met_gini3)

print()

print("Training Accuracy for tree_model_Gini_met:\n",tree_model_met_gini_training3)

# Decision Tree model # Entropy



tree_model_us_entropy3 = DecisionTreeClassifier(criterion="entropy")  

tree_model_us_entropy3.fit(X_train, y_train)



tree_model_scaled_entropy3 = tree_model_us_entropy3.predict(X_test) 

tree_model_met_entropy3=metrics.accuracy_score(y_test,tree_model_scaled_entropy3)

tree_model_met_entropy_training3=tree_model_us_entropy3.score(X_train, y_train)



print("Test Accuracy for tree_model_Entropy_met:\n",tree_model_met_entropy3)

print()

print("Training Accuracy for tree_model_Entropy_met:\n",tree_model_met_entropy_training3)

# KNN model

KNN_model_us3 = KNeighborsClassifier()  

KNN_model_us3.fit(X_train, y_train)



KNN_model_scaled3 = KNN_model_us3.predict(X_test) 

KNN_model_met3=metrics.accuracy_score(y_test,KNN_model_scaled3)

KNN_model_met_training3=KNN_model_us3.score(X_train, y_train)





print("Test Accuracy for KNN_model_met:\n",KNN_model_met3)

print()

print("Training Accuracy for KNN_model_met:\n",KNN_model_met_training3)

# Naive model



Naive_model_us3 = GaussianNB()  

Naive_model_us3.fit(X_train, y_train)



Naive_model_scaled3 = Naive_model_us3.predict(X_test) 

Naive_model_met3=metrics.accuracy_score(y_test,Naive_model_scaled3)

Naive_model_met_training3=Naive_model_us3.score(X_train, y_train)





print("Test Accuracy for Naive_model_met:\n",Naive_model_met3)

print()

print("Training Accuracy for Naive_model_met:\n",Naive_model_met_training3)

# Naive model



Naive_model_us_ber3 = BernoulliNB()  

Naive_model_us_ber3.fit(X_train, y_train)



Naive_model_unscaled_ber3 = Naive_model_us_ber3.predict(X_test) 

Naive_model_met_ber3=metrics.accuracy_score(y_test,Naive_model_unscaled_ber3)

Naive_model_met_ber_tr3=Naive_model_us_ber3.score(X_train, y_train)





print("Test Accuracy for Naive_model_met_bernouli:\n",Naive_model_met_ber3)

print()

print("Training Accuracy for Naive_model_met_bernouli:\n",Naive_model_met_ber_tr3)

# SVM model



SVM_model_us3 = SVC()  

SVM_model_us3.fit(X_train, y_train)



SVM_model_unscaled3 = SVM_model_us3.predict(X_test) 

SVM_model_met3=metrics.accuracy_score(y_test,SVM_model_unscaled3)

SVM_model_met_tr3=SVM_model_us3.score(X_train, y_train)





print("Test Accuracy for SVM_model_met:\n",SVM_model_met3)

print()

print("Training Accuracy for SVM_model_met:\n",SVM_model_met_tr3)

Accuracy_Scores_5=pd.DataFrame([y_random_met3,y_log_model_met3,tree_model_met_gini3,tree_model_met_entropy3,

                                     KNN_model_met3,Naive_model_met3,Naive_model_met_ber3,SVM_model_met3],columns=["Test_Accuracy_SL_PCA"]

                               ,index= ["Random_met","Log_model_met","tree_model_met_gini","tree_model_met_entropy",

                                     "KNN_model_met","Naive_model_met","Naive_model_met_ber","SVM_model_met"])



Accuracy_Scores_5.sort_values(by="Test_Accuracy_SL_PCA",ascending=False)



Accuracy_Scores_5["Training_Accuracy_SL_PCA"] = [y_random_met_training3,y_log_model_met_training3,

                                          tree_model_met_gini_training3,tree_model_met_entropy_training3,

                                     KNN_model_met_training3,Naive_model_met_training3,

                                          Naive_model_met_ber_tr3,SVM_model_met_tr3]

Accuracy_Scores_5.sort_values(by="Test_Accuracy_SL_PCA",ascending=False)
comparison = pd.concat([Accuracy_Scores_2,Accuracy_Scores_3,Accuracy_Scores_4,Accuracy_Scores_5],1)

comparison1=pd.DataFrame(comparison.T,columns=["Ran_For","Log_model","Tree_Gini","Tree_Entropy","KNN","Naive_G","Naive_B","SVM"])
j=0

for i in comparison1.columns:

    comparison1[i]=comparison.T.iloc[:,j]

    j=j+1
comparison1
models=[]

models.append(("random_us_USL_PCA",random_us))

models.append(("log_model_us_USL_PCA",log_model_us))

models.append(("tree_model_us_gini_USL_PCA",tree_model_us_gini))

models.append(("tree_model_us_entropy_USL_PCA",tree_model_us_entropy))

models.append(("KNN_model_us_USL_PCA",KNN_model_us))

models.append(("Naive_model_us_USL_PCA",Naive_model_us))

models.append(("Naive_model_us_ber_USL_PCA",Naive_model_us_ber))

models.append(("SVM_model_us_USL_PCA",SVM_model_us))



models.append(("random_us1_USL_No_PCA",random_us1))

models.append(("log_model_us1_USL_No_PCA",log_model_us1))

models.append(("tree_model_us_gini1_USL_No_PCA",tree_model_us_gini1))

models.append(("tree_model_us_entropy1_USL_No_PCA",tree_model_us_entropy1))

models.append(("KNN_model_us1_USL_No_PCA",KNN_model_us1))

models.append(("Naive_model_us1_USL_No_PCA",Naive_model_us1))

models.append(("Naive_model_us_ber1_USL_No_PCA",Naive_model_us_ber1))

models.append(("SVM_model_us1_USL_No_PCA",SVM_model_us1))



models.append(("random_us2_SL_No_PCA",random_us2))

models.append(("log_model_us2_SL_No_PCA",log_model_us2))

models.append(("tree_model_us_gini2_SL_No_PCA",tree_model_us_gini2))

models.append(("tree_model_us_entropy2_SL_No_PCA",tree_model_us_entropy2))

models.append(("KNN_model_us2_SL_No_PCA",KNN_model_us2))

models.append(("Naive_model_us2_SL_No_PCA",Naive_model_us2))

models.append(("Naive_model_us_ber2_SL_No_PCA",Naive_model_us_ber2))

models.append(("SVM_model_us2_SL_No_PCA",SVM_model_us2))



models.append(("random_us3_SL_PCA",random_us3))

models.append(("log_model_us3_SL_PCA",log_model_us3))

models.append(("tree_model_us_gini3_SL_PCA",tree_model_us_gini3))

models.append(("tree_model_us_entropy3_SL_PCA",tree_model_us_entropy3))

models.append(("KNN_model_us3_SL_PCA",KNN_model_us3))

models.append(("Naive_model_us3_SL_PCA",Naive_model_us3))

models.append(("Naive_model_us_ber3_SL_PCA",Naive_model_us_ber3))

models.append(("SVM_model_us3_SL_PCA",SVM_model_us3))
PCA_Models =[random_us,log_model_us,tree_model_us_gini,tree_model_us_entropy,KNN_model_us,

             Naive_model_us,Naive_model_us_ber,SVM_model_us,random_us3,log_model_us3,

             tree_model_us_gini3,tree_model_us_entropy3,KNN_model_us3,Naive_model_us3,

             Naive_model_us_ber3,SVM_model_us3]
USL_Models=[random_us,log_model_us,tree_model_us_gini,tree_model_us_entropy,KNN_model_us,

             Naive_model_us,Naive_model_us_ber,SVM_model_us,random_us1,log_model_us1,tree_model_us_gini1,

            tree_model_us_entropy1,KNN_model_us1,Naive_model_us1,Naive_model_us_ber1,SVM_model_us1]
y_usl = y_kmean.copy()

y_sl = Y
# Evaluate each model in turn

from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve,auc

from sklearn import model_selection

kfold = model_selection.KFold(n_splits=5,random_state=0,shuffle=True)



var_Err = []

Bias_Err = []

Auc_Scr = []



for name,modelss in models:

    

    roc_auc = []

    

    if modelss in USL_Models:

        Y = np.array(y_usl)

    else:

        Y = np.array(y_sl)

    

    for train,test in kfold.split(X,Y):

        

        xtrain,xtest=X.iloc[train,:],X.iloc[test,:]

        if modelss in PCA_Models:

            pca = PCA(n_components=2)  

            xtrain = pca.fit_transform(xtrain)  

            xtest = pca.transform(xtest)

            

        ytrain,ytest=Y[train],Y[test]

        modelss.fit(xtrain,ytrain)

        y_predict=modelss.predict(xtest)

        

        fpr,tpr,_ = roc_curve(ytest,y_predict)

        roc_auc.append(auc(fpr,tpr))

    var = "%0.5f"%np.var(roc_auc,ddof=1)

    bias= "%0.03f"%(1-np.mean(roc_auc))

    auc2 = "%0.03f"%np.mean(roc_auc)

    var_Err.append(var)

    Bias_Err.append(bias)

    Auc_Scr.append(auc2)

    

    

    print("AUC scores: %0.03f (+/- %0.5f) [%s]" % (np.mean(roc_auc),np.var(roc_auc,ddof=1),name))



    
Score = pd.DataFrame({"AUC_Score":Auc_Scr,"Bias_Error":Bias_Err,"Variance_Error":var_Err},index=["random_us_USL_PCA",

"log_model_us_USL_PCA",

"tree_model_us_gini_USL_PCA",

"tree_model_us_entropy_USL_PCA",

"KNN_model_us_USL_PCA",

"Naive_model_us_USL_PCA",

"Naive_model_us_ber_USL_PCA",

"SVM_model_us_USL_PCA",

"random_us1_USL_No_PCA",

"log_model_us1_USL_No_PCA",

"tree_model_us_gini1_USL_No_PCA",

"tree_model_us_entropy1_USL_No_PCA",

"KNN_model_us1_USL_No_PCA",

"Naive_model_us1_USL_No_PCA",

"Naive_model_us_ber1_USL_No_PCA",

"SVM_model_us1_USL_No_PCA",

"random_us2_SL_No_PCA",

"log_model_us2_SL_No_PCA",

"tree_model_us_gini2_SL_No_PCA",

"tree_model_us_entropy2_SL_No_PCA",

"KNN_model_us2_SL_No_PCA",

"Naive_model_us2_SL_No_PCA",

"Naive_model_us_ber2_SL_No_PCA",

"SVM_model_us2_SL_No_PCA",

"random_us3_SL_PCA",

"log_model_us3_SL_PCA",

"tree_model_us_gini3_SL_PCA",

"tree_model_us_entropy3_SL_PCA",

"KNN_model_us3_SL_PCA",

"Naive_model_us3_SL_PCA",

"Naive_model_us_ber3_SL_PCA",

"SVM_model_us3_SL_PCA"])

Score.sort_values(by="Variance_Error")
for i in Score.columns:

    Score[i]=Score[i].astype(float)
Score2=Score[Score.Variance_Error>0.00001][Score.AUC_Score>0.75].sort_values(by="Variance_Error")

Score2
from sklearn.ensemble import VotingClassifier

# Choose only the best models below:

stacked_model = VotingClassifier(estimators=[("Naive_model_us_ber1_USL_No_PCA",Naive_model_us_ber1),

                                             ("Naive_model_us1_USL_No_PCA",Naive_model_us1),

                                             ("Naive_model_us_ber_USL_PCA",Naive_model_us_ber)])

array = telecom.values 
list1=list(range(2))

list2=list(range(3,31))

X=array[:,list1+list2]

Y=array[:,2]
kfold=model_selection.KFold(n_splits=10,random_state=7)

results=model_selection.cross_val_score(stacked_model,X,Y,cv=kfold)

print(results.mean())