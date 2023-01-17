from shutil import copyfile
copyfile(src = "../input/fairness.py", dst = "../working/fairness.py")
import fairness
import pandas as pd
data = pd.read_csv("../input/resumes_development.csv", index_col=0)
data.head()
skill = "Digital Media"
accepted = data.query("Interview == 1")[skill]
rejected = data.query("Interview == 0")[skill]
print("Accepted candidates with skill = {0:.1f}%".format(accepted.mean() * 100))
print("Rejected candidates with skill = {0:.1f}%".format(rejected.mean() * 100))
data[["Digital Media", "Team Management", "Java", "Analytical Skills", "Troubleshooting"]].corr().round(2)
from sklearn.model_selection import train_test_split

data_train, data_test = train_test_split(data, test_size=0.25, stratify=data["Interview"])
y_train = data_train["Interview"]
y_test = data_test["Interview"]
print("Train: N = {} records, P(Interview) = {}".format(len(data_train), y_train.mean()))
print("Test:  N = {} records, P(Interview) = {}".format(len(data_test), y_test.mean()))
demographics = ["Veteran", "Female", "URM", "Disability"]
predictors = list(set(data.columns) - set(["Interview"] + demographics))
X_train = data_train[predictors]
X_test = data_test[predictors]
from sklearn.tree import DecisionTreeClassifier
from fairness import evaluate_model, visualize_tree
tree = DecisionTreeClassifier(max_depth=7)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
evaluate_model(y_test, y_pred)
visualize_tree(tree, predictors, ["Rejected", "Interview"], ["red", "purple"])
from sklearn.linear_model import LogisticRegression
logres = LogisticRegression(solver="liblinear")
logres.fit(X_train, y_train)
y_pred = logres.predict_proba(X_test)[:,1]
evaluate_model(y_test, logres.predict(X_test))
y_modified = (logres.predict_proba(X_test)[:,1] >= 0.4).astype(int)
evaluate_model(y_test, y_modified)
from sklearn.metrics import precision_recall_curve, average_precision_score

def show_prc(clf, X_test, y_test):
    y_score = clf.predict_proba(X_test)[:, 1]
    precs, recs, prc_t = precision_recall_curve(y_test, y_score)
    print("AUPRC = {0:.3f}".format(average_precision_score(y_test, y_score)))
    best_f1 = -1
    best_t = 2
    for p, r, t in zip(precs, recs, prc_t):
        f1 = (2 * p * r) / (p + r + 1e-10)
        print("F1 = {0:.3f} at threshold = {1:.3f}".format(f1, t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print("Best F1 = {0:.3f} at threshold = {1:.3f}".format(best_f1, best_t))
    sns.lineplot(x=recs, y=precs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
from fairness import show_roc, show_prc
show_prc(logres, X_test, y_test)
coefs = pd.DataFrame(logres.coef_[0], index=predictors)
coefs.head(3)
import numpy as np
np.exp(1.176442)
