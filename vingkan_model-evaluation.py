from sklearn.metrics import confusion_matrix

def evaluate_model(y_true, y_pred, verbose=True, show_proportion=False):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    matrix = [[tp, fn], [fp, tn]]
    zero = 1e-10
    if show_proportion:
        n = tn + fp + fn + tp + zero
        matrix = [[tp / n, fn / n], [fp / n, tn / n]]
    prec = tp / (tp + fp + zero)
    rec = tp / (tp + fn + zero)
    f1 = (2 * prec * rec) / (prec + rec + zero)
    out = ""
    out += "F1 Score = {0:.3f}".format(f1)
    out += "\nPrecision = {0:.3f}".format(prec)
    out += "\nRecall = {0:.3f}".format(rec)
    table = pd.DataFrame(
        matrix, columns=["Predicted +", "Predicted -"], index=["Actual +", "Actual -"]
    )
    if verbose:
        print(out)
        return table
    else:
        return f1, prec, rec, table, (tn, fp, fn, tp)
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def show_roc(clf, X_test, y_test):
    y_score = clf.predict_proba(X_test)[:,1]
    fpr, tpr, roc_t = roc_curve(y_test, y_score)
    print("AUROC = {0:.3f}".format(roc_auc_score(y_test, y_score)))
    sns.lineplot(x=fpr, y=tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

def show_prc(clf, X_test, y_test):
    y_score = clf.predict_proba(X_test)[:,1]
    precs, recs, prc_t = precision_recall_curve(y_test, y_score)
    print("AUPRC = {0:.3f}".format(average_precision_score(y_test, y_score)))
    best_f1 = -1
    best_t = 2
    for p, r, t in zip(precs, recs, prc_t):
        f1 = (2 * p * r) / (p + r + 1e-10)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print("Best F1 = {0:.3f} at threshold = {1:.3f}".format(best_f1, best_t))
    sns.lineplot(x=recs, y=precs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
from sklearn import tree as sklearn_tree
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

def visualize_tree(model, feature_names, class_names, fill_colors):
    dot_data = StringIO()
    sklearn_tree.export_graphviz(
        model,
        out_file=dot_data,
        feature_names=feature_names,
        class_names=class_names
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    leaves = set()
    non_leaves = set()
    for edge in graph.get_edges():
        leaves.add(int(edge.get_destination()))
        non_leaves.add(int(edge.get_source()))
    leaves.difference_update(non_leaves)
    leaves_idx = list(filter(lambda l: type(l) is int, leaves))
    nodes = graph.get_nodes()
    terminal_nodes = [nodes[i + 1] for i in leaves_idx]
    for node in terminal_nodes:
        text = node.get_label()
        if text:
            cls = text[1:-1].split("nclass = ")[1]
            node.set("style", "filled")
            if cls == class_names[0]:
                node.set("fillcolor", fill_colors[0])
            elif cls == class_names[1]:
                node.set("fillcolor", fill_colors[1])
    return Image(graph.create_png())
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("hls")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
target = "high_value"
reg_target = "median_house_value"
raw = pd.read_csv("../input/housing_data.csv", index_col=0)
print("N = {} records, {} records with missing values".format(len(raw), len(raw) - len(raw.dropna())))
raw.head()
raw["ocean_proximity"] = ["within_hour_ocean" if val == "<1H OCEAN" else val for val in raw["ocean_proximity"]]
raw["ocean_proximity"] = raw["ocean_proximity"].apply(lambda x: x.lower().replace(" ", "_"))
raw = pd.get_dummies(raw, columns=["ocean_proximity"])
from sklearn.impute import SimpleImputer

mean_imp = SimpleImputer(strategy="mean")
raw["total_bedrooms_mean_imp"] = mean_imp.fit_transform(raw["total_bedrooms"].values.reshape(-1, 1))
sf_lat = 37.756460
sf_lon = -122.442749
sf_dist = np.sqrt((raw["latitude"] - sf_lat) ** 2 + (raw["longitude"] - sf_lon) ** 2)
raw["proximity_sf"] = sf_dist.max() - sf_dist

la_lat = 34.121552
la_lon = -118.360661
la_dist = np.sqrt((raw["latitude"] - la_lat) ** 2 + (raw["longitude"] - la_lon) ** 2)
raw["proximity_la"] = la_dist.max() - la_dist
raw["room_bedroom_diff"] = raw["total_rooms"] - raw["total_bedrooms"]
from sklearn.preprocessing import StandardScaler

cont_features = [
    "housing_median_age",
    "median_income",
    "population",
    "total_bedrooms_mean_imp",
    "total_rooms",
    "households",
    "proximity_sf",
    "proximity_la"
]
for feature in cont_features:
    scaler = StandardScaler()
    raw[feature + "_scaled"] = scaler.fit_transform(raw[feature].values.reshape(-1, 1))
raw.head().T
chosen_features = [
    "ocean_proximity_inland",
    "ocean_proximity_island",
    "ocean_proximity_near_bay",
    "ocean_proximity_near_ocean",
    "ocean_proximity_within_hour_ocean",
    "housing_median_age_scaled",
    "median_income_scaled",
    "population_scaled",
    "total_bedrooms_mean_imp_scaled",
    "total_rooms_scaled",
    "households_scaled",
    "proximity_sf_scaled",
    "proximity_la_scaled"
]
raw[chosen_features].head().T
from sklearn.model_selection import train_test_split

SEED = 0
data = raw.query("split == 'train'")
X = data[chosen_features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, shuffle=True, random_state=SEED
)
print("Train: N = {0} records, P({1}|X) = {2:.3f}".format(len(X_train), target, y_train.mean()))
print("Test:  N = {0} records, P({1}|X) = {2:.3f}".format(len(X_test), target, y_test.mean()))
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy="uniform")
clf.fit(X_train, y_train)
cm = evaluate_model(y_test, clf.predict(X_test))
print()
print(cm)
print()
# show_roc(clf, X_test, y_test)
# show_prc(clf, X_test, y_test)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)
cm = evaluate_model(y_test, clf.predict(X_test))
print()
print(cm)
print()
# show_roc(clf, X_test, y_test)
# show_prc(clf, X_test, y_test)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3, random_state=SEED)
clf.fit(X_train, y_train)
cm = evaluate_model(y_test, clf.predict(X_test))
print()
print(cm)
print()
# show_roc(clf, X_test, y_test)
# show_prc(clf, X_test, y_test)
CLASSES = ["Low Value", "High Value"]
FILL = ["#fca69c", "#a0c4ff"]
visualize_tree(clf, chosen_features, CLASSES, FILL)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver="liblinear", penalty="l2", max_iter=1000, random_state=SEED)
clf.fit(X_train, y_train)
cm = evaluate_model(y_test, clf.predict(X_test))
print()
print(cm)
print()
# show_roc(clf, X_test, y_test)
# show_prc(clf, X_test, y_test)
print("Intercept = {0:.3f}".format(clf.intercept_[0]))
pd.DataFrame(clf.coef_[0], index=chosen_features, columns=["Coefficient"])
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy="uniform", random_state=0)
clf.fit(X_train, y_train)

X_all = raw[chosen_features]
output = pd.DataFrame()
output["Expected"] = clf.predict(X_all)
output.head()
output_file = "submission.csv"
output.to_csv(output_file, index_label="Id")
print("Wrote predictions to {}".format(output_file))
