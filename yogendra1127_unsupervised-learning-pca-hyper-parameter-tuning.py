import numpy as np                # useful for mathematical numeric operations
import pandas as pd               # Useful for data structuring/Framing operations
import matplotlib.pyplot as plt   # Useful for data visualization
%matplotlib inline
import seaborn as sns             # Useful for data visualization
import warnings
warnings.filterwarnings('ignore')
Vehicle_df = pd.read_csv('../input/vehicle-unsupervised-learning-project/vehicle_Unsupervised Learning_project.csv')
# Check the head of Data frame

Vehicle_df.head()
# Check the tail of this Data frame

Vehicle_df.tail()
# Let us check the shape of the Data frame

print(f'Shape of the Dataframe is:- {Vehicle_df.shape}')

Total_rows = Vehicle_df.shape[0]
Total_columns = Vehicle_df.shape[1]
print("")
print(f'Total Number of rows in Data set are = {Total_rows}')
print("")
print(f'Total Number of Columns in Data set are = {Total_columns}')
# Now check the type of attributes we have in data set

Vehicle_df.info()
Vehicle_df.dtypes
# Check the presence of any missing values in this data set

Vehicle_df.isna().sum()
Vehicle_df.isnull().sum()
# Let us check the distribution of data using 5 point summary

Vehicle_df.describe().round(2).T
# Check the Value counts of Categorical attribut "Class" in this data set

print(f'''=========================================
**Summary of Class Attribute**
=========================================
{Vehicle_df['class'].value_counts()}''')
nullValues = pd.DataFrame(Vehicle_df['circularity'].isnull())

Vehicle_df[nullValues['circularity']==True] # Displays the missing value in one of the attribute
# Instead of dropping the missing value rows from data set we will replace these null values with median nummber of each attribute.

Vehicle_df.median()
# replace the missing values with median value.
# Note, we do not need to specify the column names below
# every column's missing value is replaced with that column's median respectively  (axis =0 means columnwise)

medianFiller = lambda x: x.fillna(x.median())

Vehicle_df_new = Vehicle_df.drop(['class'], axis=1) # We have to drop the categorical column from data frame to use median value in other columns
Vehicle_df_new =Vehicle_df_new.apply(medianFiller, axis=1) # Missing values are treated with median number now
class_index = Vehicle_df['class'] # Let us again add class attribute to data frame by reindexing it.

Vehicle_df_new['class'] =class_index

Vehicle_df_new.head()
# Recheck if missing values are treated and not present in new Data frame.

Vehicle_df_new.isnull().sum()
# Check in 5 point summary distribution as well

Vehicle_df_new.describe().round().T
# We will check co-relation among all attributes to understand the relationship they have.

corelation = plt.cm.viridis   # Color range used in heat map
plt.figure(figsize=(30,15))
plt.title('Corelation Between Features', y=1.02, size=20);
sns.heatmap(data=Vehicle_df_new.corr().round(2), linewidths=0.2, vmax=1, square=True, annot=True, cmap=corelation, linecolor='white');

sns.pairplot(Vehicle_df_new, diag_kind='kde')
# Explore the data distribution for "Class" column against other independent attributes using box plot

plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.title('Compactness Vs Class')
sns.boxplot(Vehicle_df_new['compactness'], Vehicle_df_new['class'], palette='Greens')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Circularity Vs Class')
sns.boxplot(Vehicle_df_new['circularity'], Vehicle_df_new['class'], palette='Reds')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Dist Circularity Vs Class')
sns.boxplot(Vehicle_df_new['distance_circularity'], Vehicle_df_new['class'], palette='Greys')

# Subplot 4
plt.figure(figsize=(23,6))
plt.subplot(1,4,1)
plt.title('Radious Ratio Vs Class')
sns.boxplot(Vehicle_df_new['radius_ratio'], Vehicle_df_new['class'], palette='Greens')

# Subplot 5
plt.subplot(1,4,2)
plt.title('pr.axis_aspect_ratio Vs Class')
sns.boxplot(Vehicle_df_new['pr.axis_aspect_ratio'], Vehicle_df_new['class'], palette='Reds')

# Subplot 6
plt.subplot(1,4,3)
plt.title('max.length_aspect_ratio Vs Class')
sns.boxplot(Vehicle_df_new['max.length_aspect_ratio'], Vehicle_df_new['class'], palette='Greys')
# Explore the data distribution for "Class" column against other independent attributes using box plot

plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.title('Scatter Ratio Vs Class')
sns.boxplot(Vehicle_df_new['scatter_ratio'], Vehicle_df_new['class'], palette='Greens')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Elongatedness Vs Class')
sns.boxplot(Vehicle_df_new['elongatedness'], Vehicle_df_new['class'], palette='Reds')

# Subplot 3
plt.subplot(1,3,3)
plt.title('pr.axis_rectangularity Vs Class')
sns.boxplot(Vehicle_df_new['pr.axis_rectangularity'], Vehicle_df_new['class'], palette='Greys')

# Subplot 4
plt.figure(figsize=(23,6))
plt.subplot(1,4,1)
plt.title('Max length rectangularity Vs Class')
sns.boxplot(Vehicle_df_new['max.length_rectangularity'], Vehicle_df_new['class'], palette='Greens')

# Subplot 5
plt.subplot(1,4,2)
plt.title('Scaled Variance Vs Class')
sns.boxplot(Vehicle_df_new['scaled_variance'], Vehicle_df_new['class'], palette='Reds')

# Subplot 6
plt.subplot(1,4,3)
plt.title('Scaled Variance 1 Vs Class')
sns.boxplot(Vehicle_df_new['scaled_variance.1'], Vehicle_df_new['class'], palette='Greys')
# Explore the data distribution for "Class" column against other independent attributes using box plot

plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.title('Scaled Radius of Gyration Vs Class')
sns.boxplot(Vehicle_df_new['scaled_radius_of_gyration'], Vehicle_df_new['class'], palette='Greens')

# Subplot 2
plt.subplot(1,3,2)
plt.title('Scaled Radius of Gyration 1 Vs Class')
sns.boxplot(Vehicle_df_new['scaled_radius_of_gyration.1'], Vehicle_df_new['class'], palette='Reds')

# Subplot 3
plt.subplot(1,3,3)
plt.title('Skewness About Vs Class')
sns.boxplot(Vehicle_df_new['skewness_about'], Vehicle_df_new['class'], palette='Greys')

# Subplot 4
plt.figure(figsize=(23,6))
plt.subplot(1,4,1)
plt.title('Skewness About 1 Vs Class')
sns.boxplot(Vehicle_df_new['skewness_about.1'], Vehicle_df_new['class'], palette='Greens')

# Subplot 5
plt.subplot(1,4,2)
plt.title('Skewness About 2 Vs Class')
sns.boxplot(Vehicle_df_new['skewness_about.2'], Vehicle_df_new['class'], palette='Reds')

# Subplot 6
plt.subplot(1,4,3)
plt.title('Hollows Ratio 1 Vs Class')
sns.boxplot(Vehicle_df_new['hollows_ratio'], Vehicle_df_new['class'], palette='Greys')
# Check the data distribution of each column using histogram
Vehicle_df_new.hist(figsize=(20,20));
Vehicle_df_new.corr().round(2).T
Vehicle_df_new.head()
# There are all numeric attributes excluding "class"

X = Vehicle_df_new.drop(['class'], axis=1) # Independent attribute selection from Data frame

y = Vehicle_df_new['class']  # Dependent variable from Data frame, though there is no such statement but we are considering it

y_scaled = pd.get_dummies(y, drop_first=True)  # As this attribute is categorical in nature hence conversion in to numeric form is required
# Let us now standardised the unit length across attributes & hence we will calculate zscore for all numeric observations.

from scipy.stats import zscore
X_scaled = X.apply(zscore)
X_scaled.head().round(2) # Top 5 reading of scaled data frame
# Import train-test model for spliting data into 70:30 ratio

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=101)
print("{0:0.2f}% Data in training Set".format(len(X_train)/len(X.index)*100))
print("")
print("{0:0.2f}% Data in testing Set".format(len(X_test)/len(X.index)*100))
# Distribution of Class attributes

print("Distribution of Cars in Original Data :{0}({1:0.2f}%)".format(len(Vehicle_df_new.loc[Vehicle_df_new['class']=='car']), (len(Vehicle_df_new.loc[Vehicle_df_new['class']=='car'])/len(Vehicle_df_new.index))*100))
print("Distribution of Buses in Original Data   :{0}({1:0.2f}%)".format(len(Vehicle_df_new.loc[Vehicle_df_new['class']=='bus']), (len(Vehicle_df_new.loc[Vehicle_df_new['class']=='bus'])/len(Vehicle_df_new.index))*100))
print("Distribution of Vans in Original Data   :{0}({1:0.2f}%)".format(len(Vehicle_df_new.loc[Vehicle_df_new['class']=='van']), (len(Vehicle_df_new.loc[Vehicle_df_new['class']=='van'])/len(Vehicle_df_new.index))*100))
# Import SVM library for Model building

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score

svc =SVC(random_state=101)
svc.fit(X_train, y_train) # SVM Model Training

print("Accuracy on training set: {:.4f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(svc.score(X_test, y_test)))
y_pred_svm = svc.predict(X_test) # Support vector machine model is ready for Predictions 

print (f'Accuracy of SVM Model is ={round(accuracy_score(y_test, y_pred_svm),4)*100}%')
print(classification_report(y_test, y_pred_svm)) # Classification Report of SVM model.
y_train_pred = svc.predict(X_train) # Prediction of Training Data
cm_svm = plt.cm.Reds_r # Color Scheme for confusion metrics
cm1 = confusion_matrix(y_train,y_train_pred, labels=['car','bus','van']) # Confusion metrix of SVM Model on Trainig Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm1_df = pd.DataFrame(cm1, columns=[i for i in ["Actual Car", "Actual Bus", "Actual Van"]], index=[i for i in ["Predict Car","Predict Bus", "Predict Van"]])
sns.heatmap(data=cm1_df, annot=True, fmt='.5g', cmap=cm_svm)

cm_svm1 = plt.cm.Blues_r # Color Scheme for confusion metrics
cm2 = confusion_matrix(y_test,y_pred_svm, labels=['car','bus','van']) # Confusion metrix of SVM Model on Testing Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm2_df = pd.DataFrame(cm2, columns=[i for i in ["Actual Car", "Actual Bus", "Actual Van"]], index=[i for i in ["Predict Car","Predict Bus", "Predict Van"]])
sns.heatmap(data=cm2_df, annot=True, fmt='.5g', cmap=cm_svm1)
# Let us first check the performance of few models using Train Test Split (70:30 Ratio)

from sklearn.linear_model import LogisticRegression  # Import logistics model from sklearn lib
from sklearn.ensemble import RandomForestClassifier  # import Random forest classifier from sklearn lib
# Logistic Regreesion Model

lr =LogisticRegression()
lr.fit(X_train,y_train)
print("Accuracy of Logistic Model is {0:0.2f}%".format (round(lr.score(X_test,y_test)*100,3)))
# Support Vector Machine Model

svc =SVC(random_state=101)
svc.fit(X_train,y_train)
print("Accuracy of SVM Model is {0:0.2f}%".format (round(svc.score(X_test,y_test)*100,3)))
# Random Forest Model

rf =RandomForestClassifier(n_estimators=40, random_state=101)
rf.fit(X_train,y_train)
print("Accuracy of Randome Forest Model is {0:0.2f}%".format (round(rf.score(X_test,y_test)*100,3)))
# Import neccessary Package for K-fold Cross validation
# Cross validation technique uses different set of data in each fold & thus gives holistic representation of model performance.  

from sklearn.model_selection import cross_val_score
K_Fold_lr = cross_val_score(LogisticRegression(), X,y).round(3) # K-fold cross validation using Logistics Regression

K_Fold_lr
# SVM with default C & gamma values

K_Fold_SVM = cross_val_score(SVC(random_state=101), X,y).round(3) # K-fold cross validation using Support Vector Machine

K_Fold_SVM
# SVM with optimized Gamma=0.0001 & C=100 Values

K_Fold_SVM1 = cross_val_score(SVC(gamma=0.0001, C=100, random_state=101), X ,y).round(3) # K-fold cross validation using Support Vector Machine

K_Fold_SVM1
# Random Forest Model with default parameter Values

K_Fold_rf = cross_val_score(RandomForestClassifier(), X,y).round(3)

K_Fold_rf
# Random Forest Model with optimized parameter Values

K_Fold_rf1 = cross_val_score(RandomForestClassifier(n_estimators=26, random_state=101), X,y).round(3)

K_Fold_rf1
print('Avg. K-Fold Score of Logistic Regression {0:0.2f}%'.format(K_Fold_lr.mean()*100))
print("")
print('Avg. K-Fold Score of SVM {0:0.2f}%'.format(K_Fold_SVM1.mean()*100))
print("")
print('Min K-Fold Score of SVM {0:0.2f}%'.format(K_Fold_SVM1.min()*100))
print("")
print('Max K-Fold Score of SVM {0:0.2f}%'.format(K_Fold_SVM1.max()*100))
print("")
print('Avg. K-Fold Score of Random Forest {0:0.2f}%'.format(K_Fold_rf.mean()*100))
# Import neccessary libraries to use PCA

from sklearn.decomposition import PCA
# Create Covariance matrix of Scaled Data now

covMatrix = np.cov(X_scaled, rowvar=False).round(2)

print(f'''Covariance Matrix of Scaled Data is Below

{covMatrix}''')
#generating the eigen values and the eigen vectors

e_value, e_vectors = np.linalg.eig(covMatrix)

print(f'''Eigen Values of given matrix are
==========================================
{e_value.round(2)}''')

print("==========================================")

print(f'''Eigen Vectors of given matrix are
==========================================
{e_vectors.round(2)}''')
# Build PCA with default variables here 18
pca = PCA(n_components=18, random_state=101)
pca.fit(X_scaled)
# Eigen Values using PCA method

print(pca.explained_variance_)
# Eigen Vectors using PCA method

print(pca.components_)
# Let us check the percentage of variance explained by each eigen value

print(pca.explained_variance_ratio_)
# Let us check the cumulative explained variance of the Data

print(f'''Cumulative Explained Variance :
==================================================================
{np.cumsum(pca.explained_variance_ratio_)*100}''')
plt.figure(figsize=(20,10))
plt.bar(list(range(1,19)), pca.explained_variance_ratio_ , alpha=0.8, align='center', color='Green', label = 'Individual explained variance')
plt.step(list(range(1,19)), np.cumsum(pca.explained_variance_ratio_), where='mid', color='Red', label = 'Cumulative explained variance')
plt.ylabel('Variation Explained', y=0.5, size=15)
plt.xlabel('Eigen Value', size=15)
plt.title('PCA plot of Variance & Eigen Values',y=1.02, size=20 )
plt.legend(loc='best')
plt.show()
# let us reduce the dimesions to 9

pca_9 = PCA(n_components=9, random_state=101)
pca_9.fit(X_scaled)
# Eigen Values after dimension reduction

print(pca_9.explained_variance_)
# Eigen Vector after dimension reduction
print(pca_9.components_)
# Let us create the new training Data after dimensionality reduction

X_pca = pca_9.transform(X_scaled) 

X_pca = pd.DataFrame(X_pca)
# Check the distribution of new data after dimension reduction

sns.pairplot(X_pca, diag_kind='kde')
# Let us split the PCA data in to training & testing in 70:30 Ratio

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=101) 
X_train_pca.head() # Showing head of the new data frame having 9 variables after dimension reduction
y_train_pca.head()
print(X_train_pca.shape) # Data after dimensionality reduction
print(X_train.shape)     # Original Data
# Let us now check the performance of support vector machine after PCA

svc_pca = SVC(gamma=0.01, C=100,random_state=101)
svc_pca.fit(X_train_pca, y_train) # SVM model is trained using training Data
y_pred_svm_pca = svc_pca.predict(X_test_pca) # SVM model is ready for predictions.
print("SVM_PCA Accuracy on training set: {:.4f}".format(svc_pca.score(X_train_pca, y_train_pca)))
print("SVM_PCA Accuracy on test set: {:.4f}".format(svc_pca.score(X_test_pca, y_test_pca)))
print (f'Accuracy of SVM Model is ={round(accuracy_score(y_test_pca, y_pred_svm_pca),4)*100}%')
print(classification_report(y_test_pca,y_pred_svm_pca))
y_train_pred_pca = svc_pca.predict(X_train_pca) # Prediction of training Data by SVM model
cm_svm_pca = plt.cm.Greys_r # Color Scheme for confusion metrics
cm3 = confusion_matrix(y_train_pca,y_train_pred_pca, labels=['car','bus','van']) # Confusion metrix of SVM Model on Trainig Data
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.title("CM of Training Data")
cm3_df = pd.DataFrame(cm3, columns=[i for i in ["Actual Car", "Actual Bus", "Actual Van"]], index=[i for i in ["Predict Car","Predict Bus", "Predict Van"]])
sns.heatmap(data=cm3_df, annot=True, fmt='.5g', cmap=cm_svm_pca)

cm_svm1_pca = plt.cm.Greens_r # Color Scheme for confusion metrics
cm4 = confusion_matrix(y_test_pca,y_pred_svm_pca, labels=['car','bus','van']) # Confusion metrix of SVM Model on Testing Data
plt.subplot(1,3,2)
plt.title("CM of Test Data")
cm4_df = pd.DataFrame(cm4, columns=[i for i in ["Actual Car", "Actual Bus", "Actual Van"]], index=[i for i in ["Predict Car","Predict Bus", "Predict Van"]])
sns.heatmap(data=cm4_df, annot=True, fmt='.5g', cmap=cm_svm1_pca)
# SVM with default C & gamma values using K-fold cross validation method

K_Fold_SVM_pca = cross_val_score(SVC(random_state=101), X_pca,y).round(3) # K-fold cross validation using Support Vector Machine

K_Fold_SVM_pca
# SVM with optimized C & gamma values using K-fold cross validation method

K_Fold_SVM_pca1 = cross_val_score(SVC(gamma=0.01, C=100,random_state=101), X_pca,y).round(3) # K-fold cross validation using Support Vector Machine

K_Fold_SVM_pca1
print(f'''Avg. K-Fold Cross Validation Score of SVM Model with Optimized parameter Values on PCA Data

{round(K_Fold_SVM_pca1.mean(),4)*100}%
==========================================================================================
Min Cross Validation Score of SVm Model
{round(K_Fold_SVM_pca1.min(),4)*100}%
==========================================================================================
{round(K_Fold_SVM_pca1.max(),4)*100}%''')
# Let us check the distribution of Data in Independent variables using box plot

X.boxplot(figsize=(20,10))
data = X  # Where X is Dataframe of independent attributes (Target column 'class' in dropped from this)
       
def replace(group):
    median, std = group.median(), group.std()  #Get the median and the standard deviation of every group 
    outliers = (group - median).abs() > 2*std # Subtract median from every member of each group. Take absolute values > 2std
    group[outliers] = group.median()       
    return group

X_Outliers = (data.transform(replace)) # Independent variables after outlier replaced by median values.
# Let us recheck the outliers presence in independent Parameters

X_Outliers.boxplot(figsize=(20,10))
# Let us now standardised the unit length across attributes & hence we will calculate zscore for all numeric observations.

from scipy.stats import zscore

X_scaled_Outliers = X_Outliers.apply(zscore)

X_scaled_Outliers.head()  # Scaled Dataframe after treating Outliers
# Let us split the new data in 70:30 ratio using train-test split

X_Out_train, X_Out_test, y_Out_train, y_Out_test = train_test_split(X_scaled_Outliers, y, test_size=0.30, random_state=101)
# Let us check SVM model performance on Data where outliers are treated by median value (X_scaled_outlier Data Frame)

svc_outlier =SVC(random_state=101)
svc_outlier.fit(X_Out_train, y_Out_train) # SVM Model Training on Outlier treated Data

print("Accuracy on training set: {:.4f}".format(svc_outlier.score(X_Out_train, y_Out_train)*100))
print("Accuracy on test set: {:.4f}".format(svc_outlier.score(X_Out_test, y_Out_test)*100))
y_pred_svm_Out = svc_outlier.predict(X_Out_test)

print("Accuracy of SVM on test Data is {:.3f}%".format(round(accuracy_score(y_Out_test, y_pred_svm_Out),3)*100))
# Confusion Matrix

print(confusion_matrix(y_Out_test, y_pred_svm_Out))
# SVM Performance using K-Fold Cross validation Method on Outlier treated Data

K_Fold_SVM_Out = cross_val_score(SVC(gamma=0.1, C=10, random_state=101), X_scaled_Outliers ,y).round(3) # K-fold cross validation using Support Vector Machine

K_Fold_SVM_Out
print('Avg. K-Fold Score of SVM after removing Outliers {0:0.2f}%'.format(K_Fold_SVM_Out.mean()*100))
print("")
print('Max K-Fold Score of SVM  after removing Outliers {0:0.2f}%'.format(K_Fold_SVM_Out.max()*100))
# Build PCA with default variables here 18

pca_Out = PCA(n_components=18, random_state=101)
pca_Out.fit(X_scaled_Outliers)
# Eigen Values using PCA method

print(pca_Out.explained_variance_)
# Eigen Vectors using PCA method

print(pca_Out.components_)
# Let us check the cumulative explained variance of the Data

print(f'''Cumulative Explained Variance :
==================================================================
{np.cumsum(pca_Out.explained_variance_ratio_)*100}''')
# let us reduce the dimesions to 9 (on Outliers removed Data)

pca_Out_9 = PCA(n_components=9, random_state=101)
pca_Out_9.fit(X_scaled_Outliers)
# Let us create the new training Data after dimensionality reduction

X_pca_Out = pca_Out_9.transform(X_scaled_Outliers) 

X_pca_Out = pd.DataFrame(X_pca_Out)
# Let us split the Outlier removed PCA data in to training & testing in 70:30 Ratio

X_train_pca_Out, X_test_pca_Out, y_train_pca_Out, y_test_pca_Out = train_test_split(X_pca_Out, y, test_size=0.3, random_state=101)
# Let us now check the performance of support vector machine after PCA (dimensionality Reduction) using Train-Test Split

svc_outlier_pca = SVC(gamma=0.2, C=10,random_state=101)
svc_outlier_pca.fit(X_train_pca_Out, y_train_pca_Out) # SVM model is trained using training Data
print("SVM_PCA Accuracy on training set after Dimension & Outlier Removal: {:.4f}".format(svc_outlier_pca.score(X_train_pca_Out, y_train_pca_Out)*100))
print("SVM_PCA Accuracy on test set after Dimension & Outlier Removal: {:.4f}".format(svc_outlier_pca.score(X_test_pca_Out, y_test_pca_Out)*100))
y_pred_svm_pca_Out = svc_outlier_pca.predict(X_test_pca_Out) # Model is ready to predict the outcome now
# Now lets check the performance of SVM using K-fold cross validation method on outlier removed Data

K_Fold_SVM_pca_Out = cross_val_score(SVC(gamma=0.2, C=10,random_state=101), X_pca_Out,y).round(3) # K-fold cross validation using Support Vector Machine

K_Fold_SVM_pca_Out
# Import neccessary Libraries to compare the performance of the models

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# We will create pandas data frame for each model for performance evaluation.

Performance_df1 = pd.DataFrame({'Method':['Train-Test Split'],'Model Name':['SVM on Raw Data'], 'Testing Data Accuracy':[accuracy_score(y_test,y_pred_svm)],
                              'Recall (Min)':[recall_score(y_test,y_pred_svm, average=None).mean()], 'Precision (Max)':[precision_score(y_test,y_pred_svm, average=None).mean()],
                              'F1-Score':[f1_score(y_test,y_pred_svm, average=None).mean()]})

Performance_df2 = pd.DataFrame({'Method':['Train-Test Split'],'Model Name':['SVM after PCA'], 'Testing Data Accuracy':[accuracy_score(y_test_pca,y_pred_svm_pca)],
                              'Recall (Min)':[recall_score(y_test_pca,y_pred_svm_pca, average=None).mean()], 'Precision (Max)':[precision_score(y_test_pca,y_pred_svm_pca, average=None).mean()],
                              'F1-Score':[f1_score(y_test_pca,y_pred_svm_pca, average=None).mean()]})

Performance_df3 = pd.DataFrame({'Method':['Train-Test Split'],'Model Name':['SVM After Outlier Removal'], 'Testing Data Accuracy':[accuracy_score(y_Out_test,y_pred_svm_Out)],
                              'Recall (Min)':[recall_score(y_Out_test,y_pred_svm_Out, average=None).mean()], 'Precision (Max)':[precision_score(y_Out_test,y_pred_svm_Out, average=None).mean()],
                              'F1-Score':[f1_score(y_Out_test,y_pred_svm_Out, average=None).mean()]})


Performance_df4 = pd.DataFrame({'Method':['Train-Test Split'],'Model Name':['SVM after PCA & Outlier Removal'], 'Testing Data Accuracy':[accuracy_score(y_test_pca_Out,y_pred_svm_pca_Out)],
                              'Recall (Min)':[recall_score(y_test_pca_Out,y_pred_svm_pca_Out, average=None).mean()], 'Precision (Max)':[precision_score(y_test_pca_Out,y_pred_svm_pca_Out, average=None).mean()],
                              'F1-Score':[f1_score(y_test_pca_Out,y_pred_svm_pca_Out, average=None).mean()]})

                                
Performance_df5 = pd.DataFrame({'Method':['K-Fold Cross Validation'],'Model Name':['SVM on Raw Data'], 'Testing Data Accuracy':[K_Fold_SVM1.mean()],
                              'Recall (Min)':[K_Fold_SVM1.min()], 'Precision (Max)':[K_Fold_SVM1.max()],'F1-Score':['-']})                                
                                                 
    
Performance_df6 = pd.DataFrame({'Method':['K-Fold Cross Validation'],'Model Name':['SVM after PCA'], 'Testing Data Accuracy':[K_Fold_SVM_pca1.mean()],
                              'Recall (Min)':[K_Fold_SVM_pca1.min()], 'Precision (Max)':[K_Fold_SVM_pca1.max()],'F1-Score':['-']})

Performance_df7 = pd.DataFrame({'Method':['K-Fold Cross Validation'],'Model Name':['SVM After Outlier Removal'], 'Testing Data Accuracy':[K_Fold_SVM_Out.mean()],
                              'Recall (Min)':[K_Fold_SVM_Out.min()], 'Precision (Max)':[K_Fold_SVM_Out.max()],'F1-Score':['-']})  

Performance_df8 = pd.DataFrame({'Method':['K-Fold Cross Validation'],'Model Name':['SVM after PCA & Outlier Removal'], 'Testing Data Accuracy':[K_Fold_SVM_pca_Out.mean()],
                              'Recall (Min)':[K_Fold_SVM_pca_Out.min()], 'Precision (Max)':[K_Fold_SVM_pca_Out.max()],'F1-Score':['-']})
# Let us concat the above data frames for final comparision of Model performance

Concat_df = pd.concat([Performance_df1,Performance_df2, Performance_df3, Performance_df4, 
                       Performance_df5, Performance_df6, Performance_df7, Performance_df8])

Concat_df.set_index(keys=[['1', '2', '3', '4', '5', '6', '7', '8']]).round(3)
# Import neccessary package to perform the operation
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(svc_outlier_pca, cv=10, param_grid={'kernel': ('linear','poly','rbf','sigmoid'),
             'gamma': ('auto', 'scale'), 'C':[1,10,20,30,40,50,60,70,80,90,100]})
gs.fit(X_train_pca_Out,y_train_pca_Out) # Griedsearch is trained using training Data to identify the best set of Parameters for SVM Model
gs.best_params_  # This command will give us best set of parameters to optimize the model performance
gs.cv_results_['params'] # Will print list of all parameters in the model
gs.cv_results_['mean_test_score'] # This operation will give the result of SVM Model for each set of parameters