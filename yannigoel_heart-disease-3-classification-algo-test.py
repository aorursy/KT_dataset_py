import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.shape
df["target"].value_counts()
df["target"].value_counts().plot(kind="bar", color=["lightblue","salmon"])
df.info()
df.isna().sum()
df.describe()
df.sex.value_counts()
pd.crosstab(df.sex,df.target)
pd.crosstab(df.target, df.sex).plot(kind="bar",
                                  figsize=(6,4),
                                  color=["salmon","lightblue"])

plt.title("Heart Disease Frequency for sex")
plt.xlabel("0 = No Disease , 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"])
plt.xticks(rotation=0)
df["thalach"].value_counts()
plt.figure(figsize=(8,6))

plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           c='salmon')
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           c='lightblue')

plt.title("Heart Disease in function of age and max Heart rate")
plt.xlabel("age")
plt.ylabel("Max heart rate")
plt.legend(["Disease","No Disease"])
df.age.plot.hist()
df.cp.value_counts().plot(kind="bar")
pd.crosstab(df.cp,df.target)
pd.crosstab(df.cp,df.target).plot(kind="bar",
                                 figsize=(6,4),
                                color=["salmon","lightblue"])

plt.title("Heart Disease frequency Per chest pain type")
plt.xlabel("Chest pain type")
plt.ylabel("amount")
plt.legend(["No Disease","Disease"])
plt.xticks(rotation=0)
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10,8))
ax = sns.heatmap(corr_matrix,
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                cmap="YlGnBu")
X = df.drop("target", axis=1)
y = df["target"] 

X
y
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_train.head()
X_test.tail()
y_train

y_test
model = {"Logistic Regression": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier()}

def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and train ML models.
    model: a dict of Diferent Scikit-learn models 
    X_train: training data (no labels)
    X_test: test data (no labels)
    y_train: train labels
    y_test: test labels
    
    """
    np.random.seed(42)
    
    model_scores = {}
    for name, model in models.items():
        
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
model_scores = fit_and_score(models=model,
                     X_train=X_train,
                     X_test=X_test,
                     y_train=y_train,
                     y_test=y_test)
model_scores
model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar()
plt.xticks(rotation=0)
train_scores = []
test_scores = []

neighbors = range(1,21)

knn = KNeighborsClassifier()

for i in neighbors: 
    knn.set_params(n_neighbors=i)
    
    knn.fit(X_train, y_train)
    
    train_scores.append(knn.score(X_train, y_train))
    
    test_scores.append(knn.score(X_test, y_test))
    
plt.plot(neighbors, train_scores, label="Train Score")
plt.plot(neighbors, test_scores, label="Test Score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbor")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data : {max(test_scores)*100:.2f}%")
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Different RandomForestClassifier hyperparameters
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}
np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model
rs_log_reg.fit(X_train, y_train);
rs_log_reg.best_params_
rs_log_reg.score(X_test, y_test)
# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model
rs_rf.fit(X_train, y_train);
rs_rf.best_params_
rs_rf.score(X_test, y_test)
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train);
gs_log_reg.best_params_
gs_log_reg.score(X_test, y_test)
y_preds = gs_log_reg.predict(X_test)
y_preds
y_test
plot_roc_curve(gs_log_reg, X_test, y_test)
print(confusion_matrix(y_test, y_preds))
sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    fig, ax = plt.subplots(figsize=(3,3)) 
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                    annot=True,
                    cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")

plot_conf_mat(y_test, y_preds)
print(classification_report(y_test, y_preds))
gs_log_reg.best_params_
clf = LogisticRegression(C=0.23357214690901212,
                         solver="liblinear")
cv_acc = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="accuracy")
cv_acc
cv_acc = np.mean(cv_acc)
cv_acc
cv_precision = np.mean(cross_val_score(clf,
                                       X,
                                       y,
                                       cv=5, # 5-fold cross-validation
                                       scoring="precision")) # precision as scoring
cv_precision
cv_recall = np.mean(cross_val_score(clf,
                                    X,
                                    y,
                                    cv=5, # 5-fold cross-validation
                                    scoring="recall")) # recall as scoring
cv_recall
cv_f1 = np.mean(cross_val_score(clf,
                                X,
                                y,
                                cv=5, # 5-fold cross-validation
                                scoring="f1")) # f1 as scoring
cv_f1
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                            "Precision": cv_precision,
                            "Recall": cv_recall,
                            "F1": cv_f1},
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);
clf.fit(X_train, y_train)
print("Train set score : {}".format(clf.score(X_train,y_train)))
print("Test set score : {}".format(clf.score(X_test,y_test)))
