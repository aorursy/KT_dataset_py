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

from sklearn.naive_bayes import GaussianNB

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
attrition = pd.read_csv("../input/IBM-HR-Employee-Churn.csv")

attrition.head()
attrition.shape  



#The dataset has 1470 samples and 35 features
null_sum = attrition.isnull().sum()

null_sum[null_sum!=0]



# There are no null values present in the data
attrition.describe().transpose()
plt.figure(figsize=(10,5))

sns.boxplot(data=attrition)

plt.xticks(rotation=90)



# By looking the box plot, we can see that there are some outliers 

# in the "Monthly Income" Feature as we expected !!!



# Since, the outliers are not so far.. instead of removing the outliers, we can perform capping !!

# Let's find the upper range and lower range for the feature "MonthlyIncome"



Q_3 = attrition.MonthlyIncome.quantile(.75)

Q_1 = attrition.MonthlyIncome.quantile(.25)



IQR = Q_3 - Q_1



upper_range = Q_3 + 1.5 * IQR



lower_range = Q_3 - 1.5 * IQR



print("Upper Range:\t",upper_range)

print("\nLower Range:\t",lower_range)

attrition.MonthlyIncome[attrition.MonthlyIncome>upper_range].count()



# There are 114 samples has range greater than upper range. We can cap them
# Cap the outliers with the use of np.where function



# np.where(condition,new_data,Old_data)



attrition.MonthlyIncome = np.where(attrition.MonthlyIncome>upper_range,

                                                          upper_range,attrition.MonthlyIncome)
attrition.MonthlyIncome[attrition.MonthlyIncome>upper_range].count()



# All the outliers are capped now. Let's check the box plot now
plt.figure(figsize=(10,5))

sns.boxplot(data=attrition)

plt.xticks(rotation=90)



# Now, there is no outliers
object_columns = list(attrition.select_dtypes(include="object").columns)



len(object_columns)



# So,There are totally 9 features which has categorical data in the form of text.

# We need to convert into numerical data
# This loop will give us the view that, if we use label encoder to convert the text to numerical

# data, what texts will be converted into what numbers. (Because when we use label encoder, it

# doesn't show which text converted to which numbers )



replace_values = {}

for i in object_columns:

    k=0

    temp=[]

    ab=list(attrition[i].unique())

    ab.sort()

    for j in ab:

        temp.append((j,k))

        k=k+1

    replace_values[i]=temp



replace_values        
# Let's convert the text features into the numerical features by the use of LabelEncoder



le = LabelEncoder()

df_categorical = attrition.select_dtypes(include=['object'])

df_categorical = df_categorical.apply(le.fit_transform)

df_categorical.head()
#Remove the original categorical(text) features and add the numerically converted features



attrition_copy = pd.DataFrame(attrition) # For backup/Reference



for m in df_categorical.columns:

    attrition.drop(m,axis=1,inplace=True)



    

attrition=pd.concat([attrition,df_categorical],axis=1)

attrition.head()
cor_df=pd.DataFrame(attrition.corr()["Attrition"])

cor_df["Absolute_Corr"]=abs(cor_df["Attrition"])

cor_df.sort_values(by="Absolute_Corr",ascending=False)
plt.figure(figsize=(15,10))

sns.heatmap(attrition.corr(),cmap="Greens")
# Let's convert the datatype of converted numerical columns to category



for k in df_categorical.columns:

    attrition[k]=attrition[k].astype("category")



attrition.select_dtypes(include="category").columns
# Attrition vs Gender



sns.countplot(x="Attrition",data=attrition,hue="Gender")



# Gender : 0 - Female ; 1- Male



# Attrition : 0 - No ; 1- Yes
# Attrition vs MaritalStatus

sns.countplot(x="Attrition",data=attrition,hue="MaritalStatus")



# 'MaritalStatus': [('Divorced', 0), ('Married', 1), ('Single', 2)]



# Attrition : 0 - No ; 1- Yes
# Attrition vs Department

sns.countplot(x="Attrition",data=attrition,hue="Department")



# 'Department': [('Human Resources', 0),('Research & Development', 1),('Sales', 2)]



# Attrition : 0 - No ; 1- Yes
X = attrition.drop(["EmployeeCount","StandardHours","Over18","EmployeeNumber","Attrition"],axis=1)



Y = attrition["Attrition"]
# Kmeans iteration 1 till 20 cluster size



cluster_errors = []



for i in range(1,21):

    clusters = KMeans(i)

    clusters.fit(X)

    cluster_errors.append(clusters.inertia_)

    



# WSS values



clusters_df = pd.DataFrame({"Num_clusters":range(1,21),"cluster_errors":cluster_errors})



# WSS values vs Number of Clusters (Elbow graph)



sns.pointplot(x=clusters_df.Num_clusters,y=clusters_df.cluster_errors,data=clusters_df)

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



X_train, X_test, y_train, y_test = train_test_split(X,y_hier,test_size=0.3, random_state=0)  
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
# Random Forest model

random_us = RandomForestClassifier()  

random_us.fit(X_train, y_train)



y_random_unscaled = random_us.predict(X_test) 

y_random_met=metrics.accuracy_score(y_test,y_random_unscaled)



print("Accuracy for y_random_met:\n",y_random_met)
# Logistic Regression model



log_model_us = LogisticRegression()  

log_model_us.fit(X_train, y_train)



log_model_unscaled = log_model_us.predict(X_test) 

y_log_model_met=metrics.accuracy_score(y_test,log_model_unscaled)



print("Accuracy for y_log_model_met:\n",y_log_model_met)
# Decision Tree model



tree_model_us = DecisionTreeClassifier()  

tree_model_us.fit(X_train, y_train)



tree_model_unscaled = tree_model_us.predict(X_test) 

tree_model_met=metrics.accuracy_score(y_test,tree_model_unscaled)



print("Accuracy for tree_model_met:\n",tree_model_met)
# KNN model



KNN_model_us = KNeighborsClassifier()  

KNN_model_us.fit(X_train, y_train)



KNN_model_unscaled = KNN_model_us.predict(X_test) 

KNN_model_met=metrics.accuracy_score(y_test,KNN_model_unscaled)



print("Accuracy for KNN_model_met:\n",KNN_model_met)
# Naive model



Naive_model_us = GaussianNB()  

Naive_model_us.fit(X_train, y_train)



Naive_model_unscaled = Naive_model_us.predict(X_test) 

Naive_model_met=metrics.accuracy_score(y_test,Naive_model_unscaled)



print("Accuracy for Naive_model_met:\n",Naive_model_met)
# SVM model



SVM_model_us = SVC()  

SVM_model_us.fit(X_train, y_train)



SVM_model_unscaled = SVM_model_us.predict(X_test) 

SVM_model_met=metrics.accuracy_score(y_test,SVM_model_unscaled)



print("Accuracy for SVM_model_met:\n",SVM_model_met)
Accuracy_Scores_2=pd.DataFrame([y_random_met,y_log_model_met,tree_model_met,

                                     KNN_model_met,Naive_model_met,SVM_model_met],columns=["Accuracy_Scores_Unscaled_USL"]

                               ,index= ["Random_met","Log_model_met","Tree_model_met",

                                     "KNN_model_met","Naive_model_met","SVM_model_met"])



Accuracy_Scores_2.sort_values(by="Accuracy_Scores_Unscaled_USL",ascending=False)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_std = preprocessing.StandardScaler().fit_transform(X)



X_train1, X_test1, y_train1, y_test1 = train_test_split(X_std,y_hier,test_size=0.3, random_state=0)  

#PCA for all the features



pca2 = PCA()  

X_train1 = pca2.fit_transform(X_train1)  

X_test2 = pca2.transform(X_test1)  



# Loading Score

print("Loading Scores are:\n",pca2.explained_variance_ratio_ )

#Explained variance vs Number of components plot



plt.figure(figsize=(10,3))

plt.plot(pca2.explained_variance_ratio_,marker="o",c="r")

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.xticks(range(30))

plt.show()

# Random Forest model

random_sv = RandomForestClassifier()  

random_sv.fit(X_train1, y_train1)



y_random_scaled = random_sv.predict(X_test1) 

y_random_scaled_met=metrics.accuracy_score(y_test1,y_random_scaled)



print("Accuracy for y_random_scaled_met:\n",y_random_scaled_met)
# Logistic Regression model



log_model_sv = LogisticRegression()  

log_model_sv.fit(X_train1, y_train1)



log_model_scaled = log_model_sv.predict(X_test1) 

y_log_model_scaled_met=metrics.accuracy_score(y_test1,log_model_scaled)



print("Accuracy for y_log_model_scaled_met:\n",y_log_model_scaled_met)

# Decision Tree model



tree_model_sv = DecisionTreeClassifier()  

tree_model_sv.fit(X_train1, y_train1)



tree_model_scaled = tree_model_sv.predict(X_test1) 

tree_model_scaled_met=metrics.accuracy_score(y_test1,tree_model_scaled)



print("Accuracy for tree_model_scaled_met:\n",tree_model_scaled_met)
# KNN model



KNN_model_sv = KNeighborsClassifier()  

KNN_model_sv.fit(X_train1, y_train1)



KNN_model_scaled = KNN_model_sv.predict(X_test1) 

KNN_model_scaled_met=metrics.accuracy_score(y_test1,KNN_model_scaled)



print("Accuracy for KNN_model_scaled_met:\n",KNN_model_scaled_met)

# Naive model



Naive_model_sv = GaussianNB()  

Naive_model_sv.fit(X_train1, y_train1)



Naive_model_scaled = Naive_model_sv.predict(X_test1) 

Naive_model_scaled_met=metrics.accuracy_score(y_test1,Naive_model_scaled)



print("Accuracy for Naive_model_scaled_met:\n",Naive_model_scaled_met)

# SVM model



SVM_model_sv = SVC()  

SVM_model_sv.fit(X_train1, y_train1)



SVM_model_scaled = SVM_model_sv.predict(X_test1) 

SVM_model_scaled_met=metrics.accuracy_score(y_test1,SVM_model_scaled)



print("Accuracy for SVM_model_scaled_met:\n",SVM_model_scaled_met)

Accuracy_Scores_3=pd.DataFrame([y_random_scaled_met,y_log_model_scaled_met,tree_model_scaled_met,

                                     KNN_model_scaled_met,Naive_model_scaled_met,SVM_model_scaled_met],columns=["Accuracy_Scores_scaled_USL"]

                               ,index= ["Random_scaled_met","Log_model_scaled_met","Tree_model_scaled_met",

                                     "KNN_model_scaled_met","Naive_model_scaled_met","SVM_model_scaled_met"])



Accuracy_Scores_3.sort_values(by="Accuracy_Scores_scaled_USL",ascending=False)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_train2, X_test2, y_train2, y_test2 = train_test_split(X,Y,test_size=0.3, random_state=0)  

# Random Forest model

random_sup = RandomForestClassifier()  

random_sup.fit(X_train2, y_train2)



y_random_sup = random_sup.predict(X_test2) 

y_random_sup_met=metrics.accuracy_score(y_test2,y_random_sup)



print("Accuracy for y_random_sup_met:\n",y_random_sup_met)
# Logistic Regression model



log_model_sup = LogisticRegression()  

log_model_sup.fit(X_train2, y_train2)



y_log_model_sup = log_model_sup.predict(X_test2) 

y_log_model_sup_met=metrics.accuracy_score(y_test2,y_log_model_sup)



print("Accuracy for y_log_model_sup_met:\n",y_log_model_sup_met)
# Decision tree model



tree_model_sup = DecisionTreeClassifier()  

tree_model_sup.fit(X_train2, y_train2)



y_tree_model_sup = tree_model_sup.predict(X_test2) 

y_tree_model_sup_met=metrics.accuracy_score(y_test2,y_tree_model_sup)



print("Accuracy for tree_model_scaled_met:\n",y_tree_model_sup_met)
# KNN model



KNN_model_sup = KNeighborsClassifier()  

KNN_model_sup.fit(X_train2, y_train2)



y_KNN_model_sup = KNN_model_sup.predict(X_test2) 

y_KNN_model_sup_met=metrics.accuracy_score(y_test2,y_KNN_model_sup)



print("Accuracy for KNN_model_scaled_met:\n",y_KNN_model_sup_met)



# Naives model



Naives_model_sup = GaussianNB()  

Naives_model_sup.fit(X_train2, y_train2)



y_Naives_model_sup = Naives_model_sup.predict(X_test2) 

y_Naives_model_sup_met=metrics.accuracy_score(y_test2,y_Naives_model_sup)



print("Accuracy for Naives_model_scaled_met:\n",y_Naives_model_sup_met)

# SVM model



SVM_model_sup = SVC()  

SVM_model_sup.fit(X_train2, y_train2)



y_SVM_model_sup = SVM_model_sup.predict(X_test2) 

y_SVM_model_sup_met=metrics.accuracy_score(y_test2,y_SVM_model_sup)



print("Accuracy for SVM_model_scaled_met:\n",y_SVM_model_sup_met)
Accuracy_Scores_4=pd.DataFrame([y_random_sup_met,y_log_model_sup_met,y_tree_model_sup_met,

                                     y_KNN_model_sup_met,y_Naives_model_sup_met,y_SVM_model_sup_met],columns=["Accuracy_Scores_SL"]

                               ,index= ["Random_sup_met","Log_model_sup_met","Tree_model_sup_met",

                                     "KNN_model_sup_met","Naives_model_sup_met","SVM_model_sup_met"])

Accuracy_Scores_4.sort_values(by="Accuracy_Scores_SL",ascending=False)

Accuracy_Scores_4["Accuracy_Scores_SL"]
Mod_Compr=pd.DataFrame({"Unsupervised_PCA_Unscaled":Accuracy_Scores_2["Accuracy_Scores_Unscaled_USL"].values,

             "Unsupervised_PCA_Scaled":Accuracy_Scores_3["Accuracy_Scores_scaled_USL"].values,

             "Supervised_Unscaled":Accuracy_Scores_4["Accuracy_Scores_SL"].values},

            index=["Random_Model","Log_Model","Tree_Model","KNN_Model","Naive_Bayes_Model",

                  "SVM_Model"])



Mod_Compr