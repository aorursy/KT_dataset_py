import pandas as pd 

import os
#df = pd.read_csv("fm_data.csv")

df = pd.read_csv(os.path.join('..','input','fm_data.csv'))
df.head()
y = df[['Label']]

y.head()
X =df[['Avg_red','Avg_green','Avg_blue','Time']]

X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.tree import DecisionTreeClassifier



clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, 

                                  random_state=101)
clf_tree.fit(X_train, y_train)
predicted = clf_tree.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(predicted, y_test)
!pip install pydotplus

import pydotplus

from sklearn.tree import export_graphviz



def tree_graph_to_png(tree, feature_names, png_file_to_save):

    tree_str = export_graphviz(tree, feature_names=feature_names, 

                                     filled=True, out_file=None)

    graph = pydotplus.graph_from_dot_data(tree_str)  

    graph.write_png(png_file_to_save)
tree_graph_to_png(tree=clf_tree, feature_names=['Avg_red','Avg_green','Avg_blue','Time'], 

                  png_file_to_save='tree_graph.png')
print(y_test[:1])
from sklearn.metrics import confusion_matrix

import pylab as plt



labels = ['1', '2','3','4']

cm = confusion_matrix(y_test, predicted)

print(cm)

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Confusion matrix of the classifier')

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
import numpy as np





def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
plot_confusion_matrix(cm,labels,title='Confusion matrix',cmap=None,normalize=True)