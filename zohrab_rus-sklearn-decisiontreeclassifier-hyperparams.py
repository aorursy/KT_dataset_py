from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', 

                              max_depth=3, 

                              random_state=10)
tree.get_params()
(

    tree.max_depth,

    tree.max_features,

    tree.max_leaf_nodes,

    tree.min_impurity_decrease,

    tree.min_impurity_split,

    tree.min_samples_leaf,

    tree.min_samples_split,

    tree.min_weight_fraction_leaf,

    tree.random_state,

    tree.criterion,

    tree.presort,

    tree.splitter,

    tree.class_weight,

)
tree.set_params(max_depth=4)
from sklearn import datasets

from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

X = iris.data[:]

y = iris.target





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
tree.fit(X_train, y_train)
tree.n_features_
tree.feature_importances_
tree.n_classes_
tree.classes_
tree.n_outputs_
tree.tree_
tree.decision_path(X_train)
print(tree.decision_path(X_train))
tree.apply(X_train)
tree.apply(X_test)
tree.predict(X_test)
tree.predict_proba(X_test)
tree.predict_log_proba(X_test)
tree.score(X_test, y_test)