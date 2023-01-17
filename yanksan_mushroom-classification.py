import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

%matplotlib inline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
display(data.head())
data.describe().to_csv('basic_statistics.csv')
data.describe()
def display_chart(attribute, bar_names = None, colors = None, hue = None):
    # size of chart
    fig, ax = plt.subplots(figsize=(10,7))
    
    if hue: 
        col_list = ['red', 'green']
        col_list_palette = sns.xkcd_palette(col_list)
        ax = sns.countplot(x=attribute, data=data, hue=hue, order=data[attribute].value_counts().index, palette=col_list_palette)

    elif colors:
        # https://xkcd.com/color/rgb/
        col_list = colors
        col_list_palette = sns.xkcd_palette(col_list)
        ax = sns.countplot(x=attribute, data=data, order=data[attribute].value_counts().index, palette=col_list_palette)
        
    else:
        ax = sns.countplot(x=attribute, data=data, order=data[attribute].value_counts().index)
        
    camel_case_attribute = attribute.replace('-', ' ').capitalize().title()
    ax.set(xlabel=camel_case_attribute, ylabel='Quantity')
    title = 'Mushroom ' + camel_case_attribute + ' Quantity'
    ax.set_title(title)

    if bar_names:
        ax.set_xticklabels(bar_names)
# bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
display_chart(attribute='cap-shape', 
              bar_names=('convex', 'flat','knobbed','bell','sunken','conical'),
              colors=['cerulean'])
# bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
display_chart(attribute='cap-shape',
              bar_names=('convex', 'flat','knobbed','bell','sunken','conical'),
              hue='class')
# fibrous=f,grooves=g,scaly=y,smooth=s
display_chart(attribute='cap-surface',
              bar_names=('scaly', 'smooth', 'fibrous', 'grooves'),
              colors=['cerulean'])
# fibrous=f,grooves=g,scaly=y,smooth=s
display_chart(attribute='cap-surface',
              bar_names=('scaly', 'smooth', 'fibrous', 'grooves'),
              hue='class')
# brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
display_chart(attribute='cap-color',
              bar_names=('brown', 'grey','red','yellow','white','buff','pink','cinnamon','green','purple'),
              colors=['brown', 'grey', 'red', 'yellow', 'light grey', 'buff', 'pink', 'cinnamon', 'green', 'purple'])
# brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
display_chart(attribute='cap-color',
              bar_names=('brown', 'grey','red','yellow','white','buff','pink','cinnamon','green','purple'),
              colors=['brown', 'grey', 'red', 'yellow', 'light grey', 'buff', 'pink', 'cinnamon', 'green', 'purple'],
              hue='class')
# bruises=t,no=f
display_chart(attribute='bruises',
              bar_names=('no bruises', 'with bruises'),
              colors=['cerulean'])
# bruises=t,no=f
display_chart(attribute='bruises',
              bar_names=('no bruises', 'with bruises'),
              hue='class')
# almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
display_chart(attribute='odor',
              bar_names=('none', 'foul', 'spicy', 'fishy', 'almond', 'anise', 'pungent', 'creosote', 'musty'),
              colors=['cerulean'])
# almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
display_chart(attribute='odor',
              bar_names=('none', 'foul', 'spicy', 'fishy', 'almond', 'anise', 'pungent', 'creosote', 'musty'),
              hue='class')
# attached=a,descending=d,free=f,notched=n
display_chart(attribute='gill-attachment',
              bar_names=('free', 'attached'),
              colors=['cerulean'])
# attached=a,descending=d,free=f,notched=n
display_chart(attribute='gill-attachment',
              bar_names=('free', 'attached'),
              hue='class')
# close=c,crowded=w,distant=d
display_chart(attribute='gill-spacing',
              bar_names=('close', 'crowded'),
              colors=['cerulean'])
# close=c,crowded=w,distant=d
display_chart(attribute='gill-spacing',
              bar_names=('close', 'crowded'),
              hue='class')
# broad=b,narrow=n
display_chart(attribute='gill-size',
              bar_names=('broad', 'narrow'),
              colors=['cerulean'])
# broad=b,narrow=n
display_chart(attribute='gill-size',
              bar_names=('broad', 'narrow'),
              hue='class')
# black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
display_chart(attribute='gill-color',
              bar_names=('buff', 'pink', 'white', 'brown', 'grey', 'chocolate', 'purple', 'black', 'red', 'yellow', 'orange', 'green'),
              colors=['buff', 'pink', 'light grey', 'brown', 'grey', 'chocolate', 'purple', 'black', 'red', 'yellow', 'orange', 'green'])
# black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
display_chart(attribute='gill-color',
              bar_names=('buff', 'pink', 'white', 'brown', 'grey', 'chocolate', 'purple', 'black', 'red', 'yellow', 'orange', 'green'),
              hue='class')
# enlarging=e,tapering=t
display_chart(attribute='stalk-shape',
              bar_names=('tapering', 'enlarging'),
              colors=['cerulean'])
# enlarging=e,tapering=t
display_chart(attribute='stalk-shape',
              bar_names=('tapering', 'enlarging'),
              hue='class')
# bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
display_chart(attribute='stalk-root',
              bar_names=('bulbous', 'missing', 'equal', 'club', 'rooted'),
              colors=['cerulean'])
# bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
display_chart(attribute='stalk-root',
              bar_names=('bulbous', 'missing', 'equal', 'club', 'rooted'),
              hue='class')
# fibrous=f,scaly=y,silky=k,smooth=s
display_chart(attribute='stalk-surface-above-ring',
              bar_names=('smooth', 'silky', 'fibrous', 'scaly'),
              colors=['cerulean'])
# fibrous=f,scaly=y,silky=k,smooth=s
display_chart(attribute='stalk-surface-above-ring',
              bar_names=('smooth', 'silky', 'fibrous', 'scaly'),
              hue='class')
# fibrous=f,scaly=y,silky=k,smooth=s
display_chart(attribute='stalk-surface-below-ring',
              bar_names=('smooth', 'silky', 'fibrous', 'scaly'),
              colors=['cerulean'])
# fibrous=f,scaly=y,silky=k,smooth=s
display_chart(attribute='stalk-surface-below-ring',
              bar_names=('smooth', 'silky', 'fibrous', 'scaly'),
              hue='class')
# brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
display_chart(attribute='stalk-color-above-ring',
              bar_names=('white', 'pink', 'grey', 'brown', 'buff', 'orange', 'red', 'cinnamon', 'yellow'),
              colors=['light grey', 'pink', 'grey', 'brown', 'buff', 'orange', 'red', 'cinnamon', 'yellow'])
# brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
display_chart(attribute='stalk-color-above-ring',
              bar_names=('white', 'pink', 'grey', 'brown', 'buff', 'orange', 'red', 'cinnamon', 'yellow'),
              hue='class')
# brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
display_chart(attribute='stalk-color-below-ring',
             bar_names=('white', 'pink', 'grey', 'brown', 'buff', 'orange', 'red', 'cinnamon', 'yellow'),
             colors=['light grey', 'pink', 'grey', 'brown', 'buff', 'orange', 'red', 'cinnamon', 'yellow'])
# brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
display_chart(attribute='stalk-color-below-ring',
              bar_names=('white', 'pink', 'grey', 'brown', 'buff', 'orange', 'red', 'cinnamon', 'yellow'),
              hue='class')
# brown=n,orange=o,white=w,yellow=y
display_chart(attribute='veil-color',
              bar_names=('white', 'brown', 'orange', 'yellow'),
              colors=['light grey', 'brown', 'orange', 'yellow'])
# brown=n,orange=o,white=w,yellow=y
display_chart(attribute='veil-color',
              bar_names=('white', 'brown', 'orange', 'yellow'),
              hue='class')
# none=n,one=o,two=t
display_chart(attribute='ring-number',
              bar_names=('one', 'two', 'none'),
              colors=['cerulean'])
# none=n,one=o,two=t
display_chart(attribute='ring-number',
              bar_names=('one', 'two', 'none'),
              hue='class')
# cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
display_chart(attribute='ring-type',
              bar_names=('pendant', 'evanescent', 'large', 'flaring', 'none'),
              colors=['cerulean'])
# cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
display_chart(attribute='ring-type',
              bar_names=('pendant', 'evanescent', 'large', 'flaring', 'none'),
              hue='class')
# black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
display_chart(attribute='spore-print-color',
              bar_names=('white', 'brown', 'black', 'chocolate', 'green', 'yellow', 'orange', 'purple', 'buff'),
              colors=['light grey', 'brown', 'black', 'chocolate', 'green', 'yellow', 'orange', 'purple', 'buff'])
# black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
display_chart(attribute='spore-print-color',
              bar_names=('white', 'brown', 'black', 'chocolate', 'green', 'yellow', 'orange', 'purple', 'buff'),
              hue='class')
# abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
display_chart(attribute='population',
              bar_names=('several', 'solitary', 'scattered', 'numerous', 'abundant', 'clustered'),
              colors=['cerulean'])
# abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
display_chart(attribute='population',
              bar_names=('several', 'solitary', 'scattered', 'numerous', 'abundant', 'clustered'),
              hue='class')
# grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
display_chart(attribute='habitat',
              bar_names=('woods', 'grasses', 'paths', 'leaves', 'urban', 'meadows', 'waste'),
              colors=['cerulean'])
# grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
display_chart(attribute='habitat',
              bar_names=('woods', 'grasses', 'paths', 'leaves', 'urban', 'meadows', 'waste'),
              hue='class')
data.isnull().sum()
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # target variable to 0 and 1
        if col_data.dtype == object and col == 'class':
            col_data = col_data.replace(['p', 'e'], [1, 0])

        # If data type is categorical, convert to dummy variables
        elif col_data.dtype == object:
            
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

data = preprocess_features(data)
print("Processed feature columns ({} total features):\n{}".format(len(data.columns), list(data.columns)))
# Extract feature columns
feature_cols = list(data.columns[1:])

# Extract target column 'class'
target_col = data.columns[0] 

# Show the list of columns
print("Feature columns:\n{}".format(feature_cols))
print("\nTarget column: {}".format(target_col))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = data[feature_cols]
y_all = data[target_col]

# Show the feature information by printing the first five rows
print("\nFeature values:")
display(X_all.head())
# Utility function to report best scores
def report(results, n_top=3, score='score'):
    print()
    print(f"## {score} ##")
    print()
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results[f"rank_test_{score}"] == i)
        for candidate in candidates:
            print("Mean development score: {0:.3f} (std: {1:.3f}) for parameters: {2}".format(
                  results[f"mean_test_{score}"][candidate],
                  results[f"std_test_{score}"][candidate],
                  results['params'][candidate]))
    print()
            
def plot_result(results, param, xfrom, xto):
    plt.figure(figsize=(10, 6))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel(param)
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    ax.set_xlim(xfrom, xto)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results[f"param_{param}"].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    
    return plt
def fit_model(X_all, y_all, X_train, y_train, X_test, y_test, model, scoring):
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20)
    cv_sets.split(X_train)

    # Create a classifier
    classifier = model.get('classifier')

    # Create the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(classifier, model.get('params'), scoring=scoring, cv=cv_sets, refit='accuracy') 

    # Fit the grid search object to the data to compute the optimal model
    train_start = time.time()
    grid = grid.fit(X_train, y_train)
    train_end = time.time()
    
    # Show scores and best params
    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()
    print("Grid results on development set:")
    print()
    
    grid_results = grid.cv_results_

    for key, value in scoring.items():
        report(grid_results, score=key)
        
    for params in model.get('params'):
        for param, value in params.items():
            if param == 'max_depth':
                xfrom = 1
                xto = 12
                plt = plot_result(grid_results, param, xfrom, xto)
                plt.show()
            elif param == 'n_neighbors':
                xfrom = 1
                xto = 7
                plt = plot_result(grid_results, param, xfrom, xto)
                plt.show()
            elif param == 'C':
                xfrom = 1
                xto = 1050
                plt = plot_result(grid_results, param, xfrom, xto)
                plt.show()
            elif param == 'gamma':
                xfrom = 0.0011
                xto = 0.0001
                plt = plot_result(grid_results, param, xfrom, xto)
                plt.show()
    
    # Show scores on test
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    
    test_start = time.time()
    y_true, y_pred = y_test, grid.predict(X_test)
    test_end = time.time()
    
    print("Accuracy score: {:.2f}".format(accuracy_score(y_true, y_pred)))
    print()
    print(classification_report(y_true, y_pred))
    print()
    
    print("Training time: {:.4f} seconds".format(train_end - train_start))
    print("Test time: {:.4f} seconds".format(test_end - test_start))
    print()
    
    return grid.best_estimator_

import warnings
warnings.filterwarnings('ignore')

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

models = [
    {
        'name': 'Support Vector Machines (SVM)',
        'classifier': svm.SVC(),
        'params': [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
        ]
    },
    {
        'name': 'Random Forests',
        'classifier': RandomForestClassifier(),
        'params': [{'max_depth': range(1, 11)}]
    },
    {
        'name': 'K-Nearest Neighbors (KNN)',
        'classifier': KNeighborsClassifier(),
        'params': [{'n_neighbors': range(1,6)}]
    },
    {
        'name': 'Naive Bayes',
        'classifier': GaussianNB(),
        'params': [{}]
    },
    {
        'name': 'Decision Tree',
        'classifier': DecisionTreeClassifier(),
        'params': [{'max_depth': range(1, 11)}]
    }
]

train_sizes = [6093]

best_estimators = []

for model in models:
    
    print("################################################")
    print("# Model %s" % model.get('name'))
    print("################################################")
    print()
    
    for train_size in train_sizes:

        # Shuffle and split the dataset into the number of training and testing points above
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size = train_size)

        # Show the results of the split
        print("################################################")
        print("# Training set has {} samples.".format(X_train.shape[0]))
        print("# Testing set has {} samples.".format(X_test.shape[0]))
        print("################################################")
        print()

        reg = fit_model(X_all, y_all, X_train, y_train, X_test, y_test, model, scoring)
        best_estimators.append(reg)

mushrooms_data = pd.read_csv('../input/new-mushroomscsv/new_mushrooms.csv')
display(mushrooms_data)

old_columns = list(data.columns[1:])

new_mushrooms_one_hot_encoding = []

for index, row in mushrooms_data.iterrows():
    col_n = 0
    new_columns = []
    
    while col_n < mushrooms_data.shape[1]:
        col_value = row[col_n]
        col_name = mushrooms_data.dtypes.index[col_n]
        new_columns.append(f"{col_name}_{col_value}")
        col_n = col_n + 1
        
    col_array = [0 for x in range(117)]
    
    for new_column in new_columns:
        
        idx = 0
        while idx < len(old_columns):
            if new_column == old_columns[idx]:
                col_array[idx] = 1
                idx = len(old_columns)
            idx = idx + 1
            
    new_mushrooms_one_hot_encoding.append(col_array)
    
mushrooms_data = new_mushrooms_one_hot_encoding

svc_model = best_estimators[0]
print("# Estimation of the Support Vector Machines (SVM) model")
for i, mushroom in enumerate(svc_model.predict(mushrooms_data)):
    print("Mushroom {} is {}".format(i+1, "poisonous" if mushroom == 1 else "edible"))

print()

random_forest_model = best_estimators[1]
print("# Estimation of the Random Forest model")
for i, mushroom in enumerate(random_forest_model.predict(mushrooms_data)):
    print("Mushroom {} is {}".format(i+1, "poisonous" if mushroom == 1 else "edible"))
        
