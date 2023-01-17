#pandas and plot tool
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#Loading dataset
wine = pd.read_csv('../input/winequality-red.csv')
#Let's check how the data is distributed
wine.head()
#Information about the data columns
wine.info()
#Check the data quality
print('wine:Columns with null value:\n', wine.isnull().sum())

#Review correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    
    _ = sns.heatmap(
        df.corr(), 
        center=0,
        cmap="Blues",
        ax=ax,
        linewidths=0.1,
        annot=True, 
        annot_kws={'fontsize':14 }
    )
    plt.title('Correlation of Features', y=1.05, size=15)

correlation_heatmap(wine)
g = sns.FacetGrid(wine, col='quality', hue='quality', col_wrap=3, height=4)
g.map(plt.scatter, 'alcohol','volatile acidity',s=150,alpha=0.8, edgecolors='w')
plt.legend()
#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (0, 6.5, 10)
group_names = ['bad', 'good']
wine['quality_bin'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 
wine['quality_bin'] = label_quality.fit_transform(wine['quality_bin'])
wine['quality_bin'].value_counts()
#Define X and Y
#Define y veriable as target
Target=['quality_bin']

#Define x variables##remove SibSp based on the correlation
x=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost: 
    XGBClassifier()    
    ]

#split dataset in cross-validation with this splitter class
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = wine[Target]

#index through MLA and save performance to table
row_index = 1
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    
    #score model with cross validation
    cv_results = model_selection.cross_validate(alg, wine[x], wine[Target], cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(wine[x], wine[Target])
    MLA_predict[MLA_name] = alg.predict(wine[x])
    
    row_index+=1

    
#print and sort table
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare