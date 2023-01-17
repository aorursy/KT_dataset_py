import arff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def plot_classifier(model, x, y, h=.02, val_classes=[], ax=False, title=''):
    # start plot
    if not ax:
        fig, ax = plt.subplots()
    
    # create meshgrid
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # plot contours
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap='plasma')
    
    # configure plot
    ax.scatter(x, y,s=20, c='y', edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
def plot_diff(model, *sets, title='Classifier accuracy'):
    fig, axarr = plt.subplots(1,len(sets),figsize=(15,5))
    models = [(axarr[i], sets[i], clone(model)) for i in range(0, len(sets))]
    for ax, df, sm in models:
        train, test = train_test_split(df, test_size=0.2)
        sm.fit(train.drop(columns=['class']), train['class'])
        plot_classifier(sm, df.X, df.Y, ax=ax)
        
        # accuracy
        validate = sm.predict(np.c_[test.X,test.Y])
        score = accuracy_score(test['class'], validate)
        ax.annotate(score, xy=(0,-10), xycoords='axes points')
    fig.text(0.5, 1, title, verticalalignment='top', horizontalalignment='center', weight='bold')
train = pd.DataFrame(
    data=[(x,y,int(c)) for x,y,c in arff.load('../input/kiwhs-comp-1-complete/train.arff')],
    columns=['X', 'Y', 'class']
)

train_new = pd.DataFrame(
    data=[(x,y,int(c)) for x,y,c in arff.load('../input/daten-zur-klassifizierung/train-skewed.arff')],
    columns=['X', 'Y', 'class']
)

plot_diff(svm.SVC(probability=True, kernel='poly'), train, train_new, title='Poly Kernel SVC')
plot_diff(svm.SVC(probability=True, kernel='poly', coef0=1), train, train_new, title='Poly Kernel SVC - Changed Coef')
plot_diff(svm.SVC(probability=True), train, train_new, title='RBF (Standard) SVC')
plot_diff(svm.SVC(probability=True, gamma=4), train, train_new, title='RBF SVC - Changed Gamma')