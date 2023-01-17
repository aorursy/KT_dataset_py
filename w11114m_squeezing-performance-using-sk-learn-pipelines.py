# Generic Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import missingno as msno
import warnings
# import mlflow
import time
from pathlib import Path

# ML Model Libraries
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelBinarizer
import xgboost as xgb
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Variables are collinear")
data_path = Path("../input")

# Raw Data
df_train = pd.read_csv(data_path / "train.csv")
df_test = pd.read_csv(data_path / 'test.csv')
df_gender_sub = pd.read_csv(data_path / 'gender_submission.csv')

list_of_df = [df_train, df_test]
list_of_df_names = ["Train", "Test"]
all_features = df_train.drop('Survived', axis='columns')
final_test = df_test
Target_feature = df_train.loc[:,'Survived']

# Train Test Split for in notebook testing without submitting results to Kaggle
X_train, X_test, y_train, y_test = train_test_split(all_features,Target_feature,test_size=0.3,random_state=42)
for i in range(2):
    print("The " + list_of_df_names[i] + " dataset has a shape of: ", list_of_df[i].shape)
df_train.head(3)
df_test.head(3)
print("Train")
print(20*'-')
print(df_train.isnull().sum())
msno.matrix(df_train.sample(250))
msno.bar(df_train)
print("Train")
print('-'*40)
print(df_train.info())
# Continuous Data Plot
def cont_plot(df, feature_name, target_name, palettemap, hue_order, feature_scale): 
    df['Counts'] = "" # A trick to skip using an axis (either x or y) on splitting violinplot
    fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
    sns.distplot(df[feature_name], ax=axis0)
    axis0.set_xlim(left=0)
    sns.violinplot(
        x=feature_name, 
        y="Counts", 
        hue=target_name, 
        hue_order=hue_order, 
        data=df,    
        palette=palettemap, 
        split=True, 
        orient='h', 
        ax=axis1
    )
    axis1.set_xlim(left=0)
    axis1.set_xticks(feature_scale)
    plt.show()
    df.drop(["Counts"], axis="columns")


# Categorical/Ordinal Data Plot
def cat_plot(df, feature_name, target_name, palettemap): 
    fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
    df[feature_name].value_counts().plot.pie(autopct='%1.1f%%',ax=axis0)
    sns.countplot(
        x=feature_name, 
        hue=target_name, 
        data=df,
        palette=palettemap,
        ax=axis1
    )
    plt.show()


survival_palette = {0: "red", 1: "green"}  # Color map for visualization
df_train.loc[:,['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Survived', ascending=False)
cat_plot(df_train, 'Pclass','Survived', survival_palette)
df_train.loc[:,['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)
cat_plot(df_train, 'Sex','Survived', survival_palette)
cont_plot(df_train.loc[:,['Age','Survived']].dropna(axis=0), 'Age', 'Survived', survival_palette, [1, 0], range(0,100,10))
df_train.loc[:,['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean().sort_values(by='Survived', ascending=False)
cat_plot(df_train, 'SibSp','Survived', survival_palette)
df_train.loc[:,['Parch', 'Survived']].groupby('Parch', as_index=False).mean().sort_values(by='Survived', ascending=False)
cat_plot(df_train, 'Parch','Survived', survival_palette)
cont_plot(df_train.loc[:,['Fare','Survived']].dropna(axis=0), 'Fare', 'Survived', survival_palette, [1, 0], range(0,601,100))
df_train['Counts'] = ""
fig, axis = plt.subplots(1,1,figsize=(10,5))
sns.violinplot(x='Fare', y="Counts", hue='Survived', hue_order=[1, 0], data=df_train,
               palette=survival_palette, split=True, orient='h', ax=axis)
axis.set_xticks(range(0,100,10))
axis.set_xlim(0,100)
plt.show()
df_train = df_train.drop(["Counts"], axis="columns")
df_train.loc[:,['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False)
cat_plot(df_train, 'Embarked','Survived', survival_palette)
df_train.Ticket.head(20)
df_train.Cabin.head(20)
colormap = plt.cm.viridis
sns.heatmap(df_train.corr(),
            linewidths=0.1, 
            vmax=1.0, 
            square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)
plt.show()
titles = set()
for name in df_train['Name']:
    # This takes each name and splits them into two lists, separating the surnames from the rest of the name.
    # Then the rest of the name is selected using list indexing and split into two lists. This time separating the honorific from the rest of the name.
    # The honorific is selected using list indexing and whitespace is stripped, resulting the cleaned honorifics.
    titles.add(name.split(',')[1].split('.')[0].strip())
print(titles)
class HonorificExtractor(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Extracts honorifics from a string type column and groups them further into:
    Mr, Miss, Mrs, Master, Scholar, Religious, Officer and Noble.
    NaN is assumed to be Mr.
    """
    
    def __init__(self, column):
        self.column = column
    
    title_dictionary = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Scholar",
        "Rev": "Religious",
        "Col": "Officer",
        "Major": "Officer",
        "Mlle": "Miss",
        "Don": "Noble",
        "Dona": "Noble",
        "the Countess": "Noble",
        "Ms": "Mrs",
        "Mme": "Mrs",
        "Capt": "Noble",
        "Lady": "Noble",
        "Sir": "Noble",
        "Jonkheer": "Noble"
    }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # This takes each name and splits them into two lists, separating the surnames from the rest of the name.
        # Then the rest of the name is selected using list indexing and split into two lists. This time separating the honorific from the rest of the name.
        # The honorific is selected using list indexing and whitespace is stripped, resulting the cleaned honorifics.
        self.X_temp = X[self.column].map(lambda name:name.split(',')[1].split('.')[0].strip())
        X['Title'] = self.X_temp.map(self.title_dictionary)
        return X
test_title = HonorificExtractor(column='Name').fit_transform(df_train)
cat_plot(test_title, 'Title','Survived', survival_palette)
print("There are:", df_train.Age.isnull().sum(), "missing age values")
grouped_median_train = df_train.groupby(["Sex","Pclass", "Embarked", "Title"], as_index=False).median()
grouped_median_train = grouped_median_train.loc[:,["Sex", "Pclass", "Embarked", "Title", "Age"]]
grouped_median_train.head(3)
grouped_median_train.loc[:, :].loc[0, 'Age']
class AgeImputer(BaseEstimator, TransformerMixin):

    """
    Custom SK-Learn Transformer.
    Groups the data by Sex, Pclass, Embarked and Title, then calculates the median.
    The missing age data is then imputed based on these conditions.
    If
    """
    
    def fit(self, X, y=None):
        self.grouped_median_train = X.groupby(['Sex','Pclass', 'Embarked', 'Title'], as_index=False).median()
        self.grouped_median_train = self.grouped_median_train.loc[:,['Sex', 'Pclass', 'Embarked', 'Title', 'Age']]
        self.median_age = X.Age.median()
        return self

    def fill_age(self, row):
        condition = (
            (self.grouped_median_train['Sex'] == row['Sex']) | (self.grouped_median_train['Sex'] is None) &
            (self.grouped_median_train['Pclass'] == row['Pclass']) | (self.grouped_median_train['Pclass'] is None) &
            (self.grouped_median_train['Title'] == row['Title']) | (self.grouped_median_train['Title'] is None) &
            (self.grouped_median_train['Embarked'] == row['Embarked']) | (self.grouped_median_train['Embarked'] is None)
        )
        
        return self.grouped_median_train.loc[condition, 'Age'].values[0]

    def transform(self, X):
        # a function that fills the missing values of the Age variable
        X['Age'] = X.apply(lambda row: self.fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
        return X.copy()
test_age = AgeImputer().fit_transform(df_train)
cont_plot(test_age.loc[:,['Age','Survived']].dropna(axis=0), 'Age', 'Survived', survival_palette, [1, 0], range(0,100,10))
class AgeBinner(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Bins ages into categorical bins.
    The bin intervals are infered by eye from the cont_plot for the Age data.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        bins = pd.IntervalIndex.from_tuples([(0, 10), (10, 30), (30, 60), (60,100)])
        X['CategoricalAge'] = pd.cut(X['Age'], bins=bins)
        return X
test_age_bin = AgeBinner().fit_transform(df_train)
cat_plot(test_age_bin, 'CategoricalAge','Survived', survival_palette)
class FareBinning(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Bins fares into categorical bins
    The bin intervals are infered by eye from the cont_plot for the Fare data.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X.Fare.fillna(X.Fare.mean(), inplace=True)
        bins = pd.IntervalIndex.from_tuples([(0, 30), (30, 90), (90,600)])
        X['CategoricalFare'] = pd.cut(X['Fare'], bins=bins)
        return X
test_fare_bin = FareBinning().fit_transform(df_train)
cat_plot(test_fare_bin, 'CategoricalFare','Survived', survival_palette)
class HasCabin(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Groups the cabins into categories based on the first letter in the cabin code.
    If a field is null it is filled with "No Assigned Cabin"
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X.Cabin = X.Cabin.str[0]
        X.Cabin = X.Cabin.fillna("U")
        return X
temp = df_train
test_cabin = HasCabin().fit_transform(temp)
cat_plot(test_cabin, 'Cabin','Survived', survival_palette)
class FamilyCreator(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Creates a new feature called FamilySize by adding together SibSp and Parch.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        return X
test_family = FamilyCreator().fit_transform(df_train)
cont_plot(test_family.loc[:,['FamilySize','Survived']].dropna(axis=0), 'FamilySize', 'Survived', survival_palette, [1, 0], range(0,15,1))
class FamilyBinner(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Creates a new feature called FamilyBin.
    Bins the families into three bins based on the magnitude of the FamilySize feature.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['Family'] = ''
        X.loc[X['FamilySize'] == 0, 'Family'] = 'alone'
        X.loc[(X['FamilySize'] > 0) & (X['FamilySize'] <= 2), 'Family'] = 'small'
        X.loc[(X['FamilySize'] > 2) & (X['FamilySize'] <= 5), 'Family'] = 'medium'
        X.loc[X['FamilySize'] > 5, 'Family'] = 'large'
        return X
test_family_bin = FamilyBinner().fit_transform(df_train)
cat_plot(test_family_bin, 'Family','Survived', survival_palette)
class IsAlone(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Engineers new feature to determine whether individual is alone on the Titanic.
    Flag = 0 or 1
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['IsAlone'] = 0
        X.loc[X['FamilySize'] == 1, 'IsAlone'] = 1
        return X
test_alone = IsAlone().fit_transform(df_train)
cat_plot(test_alone, 'IsAlone','Survived', survival_palette)
class TicketProcesser(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Engineers new feature to determine whether individual is alone on the Titanic.
    In order to reduce the number of tickets and to group similar ticket identifiers together, I have taken the first two letters of the ticket to be the ticket ID.
    """
    
    def CleanTicket(self, ticket):
        ticket = ticket.replace('.', '').replace('/', '').split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0][:2]
        else: 
            return 'XXX'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['Ticket'] = X['Ticket'].map(self.CleanTicket)
        return X
test_ticket = TicketProcesser().fit_transform(df_train)
cat_plot(test_ticket, 'Ticket','Survived', survival_palette)
class DenseTransformer(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Returns a dense array if the array is sparse.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if scipy.sparse.issparse(X) == True:
            X = X.todense()
        return X
class FeatureDropper(BaseEstimator, TransformerMixin):
    
    """
    Custom SK-learn transformer.
    Drops features which are used for feature engineering but won't be used in the model.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
#         X = X.drop([
#             "Fare",
#             'Age',
#             'SibSp',
#             'Parch',
#             'FamilySize',
#             'Cabin',
#             'IsAlone'
#         ], axis="columns")
        return X
PrePreprocessingPipe = Pipeline(
    steps=[
        ("he", HonorificExtractor(column="Name")),
        ("fc", FamilyCreator()),
        ("famb", FamilyBinner()),
        ("ia", IsAlone()),
        ("ai", AgeImputer()),
        ("ab", AgeBinner()),
        ("farb", FareBinning()),
        ("cg", HasCabin()),
        ("fd", FeatureDropper())
    ]
)
numeric_features = [
    'SibSp', 
    'Parch',
    'Age',
    'Fare',
    'FamilySize'
]

numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)
categorical_features = [
    'Embarked', 
    'Sex', 
    'Pclass', 
    'CategoricalAge', 
    'CategoricalFare', 
    'Title', 
    'Ticket', 
    'Cabin',
    'Family',
    'IsAlone'
]

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)
PreprocessingPipeline = Pipeline(
    steps=[
        ("pp", PrePreprocessingPipe),
        ("ct", ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        ))
    ]
)
RFC = RandomForestClassifier(n_estimators=50, max_features='sqrt')
Model = Pipeline(
    steps=[
        ('pp', PreprocessingPipeline),
        ('to_dense', DenseTransformer()),
        ('classifier', RFC)
    ]
)

Model = Model.fit(all_features, Target_feature)
features = pd.DataFrame()
features['importance'] = Model.get_params(deep=True)['classifier'].feature_importances_
print("There are:", len(features), "features in the raw preprocessed data.")
model = Pipeline(
    steps=[
        ('pp', PreprocessingPipeline),
        ('to_dense', DenseTransformer()),
        ('fi_selector', SelectFromModel(RFC, prefit=True))
    ]
)
train_reduced = model.transform(all_features)
test_reduced = model.transform(final_test)

print("The shape of the reduced train dataset is: ", train_reduced.shape)
print("The shape of the reduced test dataset is: ", test_reduced.shape)

print("\nTherefore there are", train_reduced.shape[1], "features after the features with the highest feature importances have been selected.")
classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True, gamma='scale'),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver='lbfgs')
]

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=2)

acc_dict = {}

for train_index, test_index in sss.split(all_features.values, Target_feature.values): 
    Xtrain, Xtest = all_features.iloc[train_index], all_features.iloc[test_index]
    ytrain, ytest = Target_feature.iloc[train_index], Target_feature.iloc[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__
        Model = Pipeline(
            steps=[
                ('pp', PreprocessingPipeline),
                ('to_dense', DenseTransformer()),
                ('classifier', clf)
            ]
        )
        Model.fit(Xtrain, ytrain)
        train_predictions = Model.predict(Xtest)
        acc = accuracy_score(ytest, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc
            
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

print("Without Feature Importances")

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
plt.show()

print(log)

acc_dict = {}

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=2)

for train_index, test_index in sss.split(train_reduced, Target_feature.values): 
    Xtrain, Xtest = train_reduced[train_index], train_reduced[test_index]
    ytrain, ytest = Target_feature.iloc[train_index], Target_feature.iloc[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
        Model = Pipeline(
            steps=[
                ('classifier', clf)
            ]
        )
        Model.fit(Xtrain, ytrain)
        train_predictions = Model.predict(Xtest)
        acc = accuracy_score(ytest, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc
            
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

print("\n\nWith Feature Importances")
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
plt.show()

print(log)
classifiers = [
#     KNeighborsClassifier(3),
    SVC(),
#     DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
#     GradientBoostingClassifier(),
#     GaussianNB(),
    LogisticRegression()
]

parameter_grid = [
#     {
#         "n_neighbors": [2, 3, 4],
#         "weights": ["uniform", "distance"],
#         "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
#         "leaf_size" : [10, 15, 20],
#     },
    {
        "C": [6, 8, 10],
        "kernel": ["linear", "rbf", "poly"],
        "shrinking": [True, False],
        "probability": [True, False],
        "gamma": [2.0, 2.5, 3.0, "scale"]
    },
#     {
#         "criterion": ["gini", "entropy"],
#         "splitter": ["best", "random"],
#         "max_features": ["auto", "sqrt", "log2", None],
#         "class_weight": ["balanced", None],
#         "presort": [True, False]
#     },
    {
        "max_depth" : [16, 18, 20],
        "n_estimators": [100, 50, 10],
        "max_features": ["sqrt", "auto", "log2"],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True, False]
    },
    {
        "n_estimators": [60, 80],
        "algorithm": ["SAMME.R"],
        "learning_rate": [1.2, 1.4]
    },
#     {
#         "loss": ["deviance", "exponential"],
#         "learning_rate": [0.08, 0.1, 0.12],
#         "n_estimators": [90, 100, 110],
#         "criterion": ["friedman_mse", "mse", "mae"],
#     },
#     {
#         "var_smoothing" : [1e-9, 2e-9]
#     },
    {
        "penalty": ["l2"],
        "dual": [False],
        "tol": [1e-5],
        "C": [30, 35, 40],
        "fit_intercept": [True, False],
        "solver": ["newton-cg"],
        "max_iter": [200, 400, 1000]
    }
]

acc_dict = {}
cv_dict = {}

run_gs = False

if run_gs:
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(train_reduced,Target_feature,test_size=0.3,random_state=42)
    for clf in range(len(classifiers)):
        
        start_time = time.time()
        cross_validation = StratifiedKFold(n_splits=10, random_state=22)
    
        grid_search = GridSearchCV(
            classifiers[clf],
            scoring="accuracy",
            param_grid=parameter_grid[clf],
            cv=cross_validation,
            verbose=0,
            n_jobs=-1
        )
        
        grid_search.fit(Xtrain, ytrain)
        model = grid_search
        parameters = grid_search.best_params_
        
        prediction=grid_search.predict(Xtest)
        print("--------------The Accuracy of the {}".format(classifiers[clf].__class__.__name__), "----------------------------")
        print('The accuracy of the', str(classifiers[clf].__class__.__name__), 'is', round(accuracy_score(prediction, ytest)*100,2))
        
        result = cross_val_score(grid_search, Xtrain, ytrain, cv=10, scoring='accuracy')
        print('The cross validated score for', str(classifiers[clf].__class__.__name__), 'is:', round(result.mean()*100,2))
        y_pred = cross_val_predict(grid_search, Xtrain, ytrain, cv=10)
        sns.heatmap(confusion_matrix(ytrain, y_pred), annot=True, fmt='3.0f', cmap="summer")
        plt.title('Confusion_matrix', y=1.05, size=15)
        plt.show()
        
        print("Classifier: {}".format(classifiers[clf].__class__.__name__))
        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))
        
        acc = round(accuracy_score(prediction, ytest)*100,2)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc
            
        del model
    
        elapsed_time = time.time() - start_time
        print("Time taken", round(elapsed_time, 2), "seconds \n")
        print("-"*40, "\n")
        
    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy')
    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
    plt.show()
            
else:
    
    parameters = [
#         {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 3, 'weights': 'distance'},
        {'C': 8, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True, 'shrinking': True},
#         {'class_weight': 'balanced', 'criterion': 'gini', 'max_features': None, 'presort': True, 'splitter': 'random'},
        {'bootstrap': False, 'max_depth': 16, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 50} ,
        {'algorithm': 'SAMME.R', 'learning_rate': 1.2, 'n_estimators': 80},
#         {'criterion': 'mse', 'learning_rate': 0.12, 'loss': 'exponential', 'n_estimators': 90},
#         {'var_smoothing': 1e-09},
        {'C': 30, 'dual': False, 'fit_intercept': True, 'max_iter': 200, 'penalty': 'l2', 'solver': 'newton-cg', 'tol': 1e-05}   
    ]
    
    estimator_names = [
#         "knc",
        "svc",
#         "dtc",
        "rfc",
        "abc",
#         "gbc",
#         "gnb",
        "lr"
    ]
    
    csv = [
#         "submission_knc.csv",
        "submission_svc.csv",
#         "submission_dtc.csv",
        "submission_rfc.csv",
        "submission_abc.csv",
#         "submission_gbc.csv",
#         "submission_gnb.csv",
        "submission_lr.csv",        
    ]
    
    estimators = []
    for clf in range(len(classifiers)):
        model = classifiers[clf].set_params(**parameters[clf])
        model.fit(train_reduced, Target_feature)
        y_predict = model.predict(test_reduced)
        df_results = pd.DataFrame({"PassengerId": final_test.PassengerId, "Survived": y_predict})
        df_results.to_csv(csv[clf], index=False)
        estimators.append((estimator_names[clf], classifiers[clf].set_params(**parameters[clf])))
                         
    ensemble = VotingClassifier(estimators=estimators, voting='hard')
    classifiers.append(ensemble)
    estimator_names.append("ensemble")
    csv_ensemble = "submission_ensemble.csv"
    
    ensemble.fit(train_reduced, Target_feature)
    y_predict = model.predict(test_reduced)
    df_results = pd.DataFrame({"PassengerId": final_test.PassengerId, "Survived": y_predict})
    df_results.to_csv(csv_ensemble, index=False)
                           
    for clf, label in zip(classifiers, estimator_names):
        scores = cross_val_score(clf, train_reduced, Target_feature, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))