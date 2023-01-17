# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import timeit

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Load the training data set
data = pd.read_csv("../input/train.csv")

#== DATA EXPLORATION
dta = data.copy()
dta['freq'] = dta.groupby('place_id')['x'].transform('count')

del data
print(dta[:5])
# import relevant general libraries
import math
import itertools

# import libraries for data visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly import tools

# import libraries for classification algorithms
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
    
# create grid and split data points assign cell index to each data point
def create_grid(a, N, datapoints, start_index=1):
    n = N*N
    modVal = N - start_index + 1 # modval is used to combine the x and y values of grid into a unique cell index
    
    celldata = np.zeros((n,), dtype=[('gridcell_index', 'i4'), ('row', 'i4'),('col', 'i4')])
    gridcell = pd.DataFrame(celldata)

    cellValuesCombo = np.asarray(list(itertools.product(range(1, int(N)+1),range(1, int(N)+1))))
    gridcell['row'] = cellValuesCombo[:,0]
    gridcell['col'] = cellValuesCombo[:,1]
    gridcell['gridcell_index'] = gridcell['row'] + (modVal*gridcell['col'])
    gridcell['gridcell_index'] = gridcell['gridcell_index'].astype(int)
    
    datapoints['gridcell_x'] = (datapoints['x']*N/a) + 1
    datapoints['gridcell_x'] = datapoints['gridcell_x'].astype(int)
    datapoints['gridcell_y'] = (datapoints['y']*N/a) + 1
    datapoints['gridcell_y'] = datapoints['gridcell_y'].astype(int)
    datapoints['gridcell_index'] = datapoints['gridcell_x'] + (modVal*datapoints['gridcell_y'])
    datapoints['gridcell_index'] = datapoints['gridcell_index'].astype(int)
    datapoints = datapoints.drop(['gridcell_x', 'gridcell_y'], axis=1, inplace=True)
    
    return gridcell
"""
# filter data points by grid cell index and frequency
def filter_data(x, y, datapoints, datapoints_full, gridcell):
    filtered_data_index = datapoints[datapoints['gridcell_index'] == gridcell.iloc[x,y]]
    filtered_data = datapoints_full[datapoints_full['row_id'].isin(filtered_data_index['row_id'])].copy()
    if('place_id' in filtered_data.columns):
        filtered_data['freq_grid'] = filtered_data.groupby('place_id')['x'].transform('count')
        filtered_data = filtered_data[filtered_data['freq_grid'] > 3].copy()
    
    return filtered_data
"""
def filter_data(index, datapoints, place_frequency, calcFreq=False, enablePrints=False):
    t0 = timeit.default_timer()
    filtered_data = datapoints[datapoints['gridcell_index'] == index].copy()
    if(enablePrints): print( "filtered_data_index time interval: ", (timeit.default_timer()-t0) )
    
    """
    t0 = timeit.default_timer()
    filtered_data = datapoints_full[datapoints_full['row_id'].isin(filtered_data_index['row_id'])].copy()
    print( "filtered_data time interval: ", (timeit.default_timer()-t0) )
    """
    
    filtered_data_used = None
    filtered_data_unused = None
    if(calcFreq):
        t0 = timeit.default_timer()
        filtered_data['freq_grid'] = filtered_data.groupby('place_id')['x'].transform('count')
        if(enablePrints): print( "filtered_data-freq_grid time interval: ", (timeit.default_timer()-t0) )
    
        t0 = timeit.default_timer()
        filtered_data_used = filtered_data[filtered_data['freq_grid'] > place_frequency].copy()
        if(enablePrints): print( "filtered_data-filtered_data time interval: ", (timeit.default_timer()-t0) )
    
        t0 = timeit.default_timer()
        filtered_data_unused = filtered_data[filtered_data['freq_grid'] <= place_frequency].copy()
        if(enablePrints): print( "filtered_data-filtered_data time interval: ", (timeit.default_timer()-t0) )
    else:
        filtered_data_used = filtered_data.copy()
    
    del filtered_data
    return filtered_data_used, filtered_data_unused

# split time value into smaller group sets
def split_time(datapoints):
    datapoints['hour'] = (datapoints['time'] / 60) % 24
    datapoints['hour'] = datapoints['hour'].astype(int)
    datapoints['weekday'] = (datapoints['time'] / (60*24)) % 7
    datapoints['weekday'] = datapoints['weekday'].astype(int)
    datapoints['month'] = (datapoints['time'] / (60*24*30)) % 12
    datapoints['month'] = datapoints['month'].astype(int)
    datapoints['year'] = datapoints['time'] / (60*24*365)
    datapoints['year'] = datapoints['year'].astype(int)
    datapoints['day'] = (datapoints['time'] / (60*24)) % 365
    datapoints['day'] = datapoints['day'].astype(int)

# visualize data
def init_visualization_params(plot3d=False, size=1):
    fig = None
    axarr = None
    halfSize = 0
    
    if plot3d:
        fig = plt.figure(figsize=(16,10))
    else:
        halfSize = int(size / 2)
        fig, axarr = plt.subplots(halfSize, halfSize) if (halfSize>1) else plt.subplots(1, 1)
        fig.tight_layout()
        
    return fig, axarr, halfSize
    
def visualize_data(datapoints, count, figure, axesInfo, plot3d=False, halfSize=0):
    #-- plotting x against y with colors for each place
    if plot3d:
        projCount = 221 + count
        ax = figure.add_subplot(projCount, projection='3d')
        ax.scatter(datapoints['x'], datapoints['y'], datapoints['hour'], c=datapoints['place_id'], linewidth=0.0)
    else:
        datapoints_sub = datapoints[['x', 'y']].copy()

        #-- reduce dimensionality of features
        x_and_y = PCA(n_components=1).fit_transform(datapoints_sub)
        datapoints['x_and_y'] = x_and_y
        #tu_std = StandardScaler().fit_transform(tu)
        #x_and_y_and_time = PCA(n_components=2).fit_transform(tu_std)

        if (halfSize>0):
            xIndex = (count % halfSize) #remainder
            yIndex = (count / halfSize) #quotient
            axesInfo[xIndex, yIndex].scatter(datapoints['x_and_y'], datapoints['time'], c=datapoints['place_id'], linewidth=0.0)#
        else:
            axesInfo.scatter(datapoints['x_and_y'], datapoints['hour'], c=datapoints['place_id'], linewidth=0.0)#
        
    del datapoints_sub
    plt.show()

def train_model(datapoints, datapoints_unused, model_type, valOnFull=False, enablePrints=False):
    X = datapoints[['x', 'y', 'hour', 'accuracy', 'day']].copy()
    y = datapoints['place_id'].copy()
    X_unused = datapoints_unused[['x', 'y', 'hour', 'accuracy', 'day']].copy()
    y_unused = datapoints_unused['place_id'].copy()

    #split sample data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    
    X_test_full = X_test.append(X_unused)
    y_test_full = y_test.append(y_unused)

    if(enablePrints): print( "X_train Length: ", len(X_train) )
    if(enablePrints): print( "y_train Length: ", len(y_train) )
    if(enablePrints): print( "X_test Length: ", len(X_test) )
    if(enablePrints): print( "y_test Length: ", len(y_test) )
    if(enablePrints): print( "X_full Length: ", len(X_unused) )
    if(enablePrints): print( "y_full Length: ", len(y_unused) )

    classifier_model = None
    score_model = 0
    useLogReg = True if (model_type == "Logistic Regression") else False
    useDTClf = True if (model_type == "Decision Tree") else False
    useRFClf = True if (model_type == "Random Forest") else False
    if useLogReg:
        logReg = LogisticRegression()
        logReg.fit(X_train, y_train)
        score_model = logReg.score(X_test_full, y_test_full) if (valOnFull) else logReg.score(X_test, y_test)
        if(enablePrints): print ("\n\nlog reg score: %.3f", score_model)
        
        """
        OVR = OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)
        if( displayPrint ):
            print ("OVR accuracy score: %.3f", OVR.score(X_test, y_test))
        
        #X = features[:500]
        #y = places[:500]
        OVO = OneVsOneClassifier(LogisticRegression()).fit(X_train, y_train)
        if( displayPrint ):
            print ("OVO accuracy score: %.3f", OVO.score(X_test, y_test))
        """
        
        classifier_model = logReg
        
    elif useDTClf:
        dtClf_model = tree.DecisionTreeClassifier()
        dtClf_model.fit(X_train, y_train)
        score_model = dtClf_model.score(X_test_full, y_test_full) if (valOnFull) else dtClf_model.score(X_test, y_test)
        if(enablePrints): print ("\n\ndecision tree classifier score: %.3f", score_model)
        
        classifier_model = dtClf_model
        
    elif useRFClf:
        rfClf_model = RandomForestClassifier(n_estimators=40, # Number of trees
                                             max_features=2,    # Num features considered
                                             oob_score=True)    # Use OOB scoring*
        rfClf_model.fit(X_train, y_train)
        score_model = rfClf_model.score(X_test_full, y_test_full) if (valOnFull) else rfClf_model.score(X_test, y_test)
        #score_model = rfClf_model.oob_score_
        if(enablePrints): print ("\n\nrandom forest classifier score: %.3f", score_model)
        if(enablePrints): print ("\n\nrandom forest classifier oob score: %.3f", rfClf_model.oob_score_)
        
        classifier_model = rfClf_model
    
    if(enablePrints): print ("\n\n score on test val: %.3f", classifier_model.score(X_test, y_test))
    if(enablePrints): print ("\n\n score on test_full val: %.3f", classifier_model.score(X_test_full, y_test_full))
    
    del X
    del y
    del X_unused
    del y_unused
    del X_train
    del y_train
    del X_test
    del y_test
    del X_test_full
    del y_test_full
    return classifier_model, score_model

def filter_gridcells(gridcell, gridcell_density, gridcell_distribution, enablePrints=False):
    gridcell_used_update = None
    gridcell_unused_update = None
	
    if('gc_density' in gridcell.columns):
        t0 = timeit.default_timer()
        gridcell_used = gridcell[gridcell['gc_density'] > gridcell_density].copy()
        if(enablePrints): print( "gc_density-gridcell_used time interval: ", (timeit.default_timer()-t0) )
    
        t0 = timeit.default_timer()
        gridcell_unused = gridcell[gridcell['gc_density'] <= gridcell_density].copy()
        if(enablePrints): print( "gc_density-gridcell_unused time interval: ", (timeit.default_timer()-t0) )
		
        if('gc_dist' in gridcell.columns):
            t0 = timeit.default_timer()
            gridcell_used_update = gridcell_used[gridcell_used['gc_dist'] > gridcell_distribution].copy()
            if(enablePrints): print( "gc_dist-gridcell_used time interval: ", (timeit.default_timer()-t0) )
    
            t0 = timeit.default_timer()
            gridcell_unused_update = gridcell_used[gridcell_used['gc_dist'] <= gridcell_distribution].copy()
            if(enablePrints): print( "gc_dist-gridcell_unused time interval: ", (timeit.default_timer()-t0) )
		
            gridcell_unused_update = gridcell_unused.append(gridcell_unused_update)
    
    del gridcell_used
    del gridcell_unused
    return gridcell_used_update, gridcell_unused_update

def train_partial_data(num_parts, part_to_train, gridcell, datapoints_train, classifiers, scores, model_name, freq, validate_on_full=True, enablePrints=False):
    n = len(gridcell)
    parts = int(n / num_parts)
    min_range = (part_to_train - 1) * parts
    max_range = part_to_train * parts
    #n = 1

    # create an array of classifier models for each grid cell index
    for i in range(min_range, max_range):
        gc_index = gridcell.iloc[i]['gridcell_index']
        fData, fData_unused = filter_data(index=gc_index, datapoints=datapoints_train, place_frequency=freq, calcFreq=True)
        split_time(fData)
        split_time(fData_unused)

        #print( "i: ", i )
        #print( "gc_index: ", gc_index )
        #print( fData['place_id'].unique() )
        #print( fData[:5] )

        clf, sc = train_model(datapoints=fData, datapoints_unused=fData_unused, model_type=model_name, valOnFull=validate_on_full)
        classifiers.append(clf)
        scores.append(sc)
		
        del fData
        del fData_unused

    #return datapoints_predict

def predict_partial_data(num_parts, part_to_predict, gridcell, datapoints_predict, classifiers, enablePrints=False):
    n = len(gridcell)
    parts = int(n / num_parts)
    min_range = (part_to_predict - 1) * parts
    max_range = part_to_predict * parts
    #n = 1

    # create an array of classifier models for each grid cell index
    for i in range(min_range, max_range):
        gc_index = gridcell.iloc[i]['gridcell_index']
        print( "gc_index: ", gc_index )
        
        t0 = timeit.default_timer()
        p_fData, p_fData_unused = filter_data(index=gc_index, datapoints=datapoints_predict, place_frequency=0, calcFreq=False)
        split_time(p_fData)
        print( "filter_data && split_time time interval: ", (timeit.default_timer()-t0) )
		
        p_X = p_fData[['x', 'y', 'hour', 'accuracy', 'day']].copy()
        
        t0 = timeit.default_timer()
        probs_y = classifiers[i].predict_proba(p_X)
        print( "predict_proba time interval: ", (timeit.default_timer()-t0) )
        
        t0 = timeit.default_timer()
        probs_order = np.argsort(probs_y, axis=1)
        print( "argsort time interval: ", (timeit.default_timer()-t0) )
        
        t0 = timeit.default_timer()
        best_n = classifiers[i].classes_[probs_order[:, -3:]]
        p_fData['place_id'] = best_n.tolist()
        p_dta_pre_merge = p_fData[['row_id', 'place_id']].copy()
        print( "misc time interval: ", (timeit.default_timer()-t0) )
        
        t0 = timeit.default_timer()
        p_fData_ref = datapoints_predict.loc[datapoints_predict['gridcell_index'] == gc_index]
        p_fData_ref.loc['place_id'] = best_n.tolist()
        #datapoints_predict = pd.merge(datapoints_predict, p_dta_pre_merge, how='left', on='row_id')
        print( "merge time interval: ", (timeit.default_timer()-t0) )
		
        del p_fData
        del p_dta_pre_merge
        del p_X

    return datapoints_predict
	
	
"""
da = dta[['row_id', 'x', 'y']].copy()
gCell = create_grid(a=10, N=100, datapoints=da)
#fData = filter_data(x=0, y=0, datapoints=da, datapoints_full=dta, gridcell=gCell)
fData = filter_data(index=101, datapoints=da, datapoints_full=dta, gridcell=gCell)
split_time(fData)

print()
print( fData[:5] )

# DATA VISUALIZATION
f, axInfo, half = init_visualization_params(plot3d=False, size=1)
visualize_data(datapoints=fData, count=0, plot3d=False, halfSize=half, figure=f, axesInfo=axInfo)

# ALGORITHM
clf_model = train_model(datapoints=fData, model_type="Logisitic Regression")
"""
print( dta[:5] )

t0 = timeit.default_timer()
gCell = create_grid(a=10, N=100, datapoints=dta)
print( "gCell create_grid time interval: ", (timeit.default_timer()-t0) )

p_dta = pd.read_csv("../input/test.csv")

t0 = timeit.default_timer()
p_gCell = create_grid(a=10, N=100, datapoints=p_dta)
print( "p_gCell create_grid time interval: ", (timeit.default_timer()-t0) )

print( dta[:5] )
print( "dta length: ", len(dta) )
print( p_dta[:5] )
print( "p_dta length: ", len(p_dta) )

p_dta['place_id'] = np.nan
clf_models = []
sc_models = []

clf_models = []
sc_models = []

start_time = timeit.default_timer()

train_partial_data(num_parts=10000, part_to_train=1, gridcell=gCell, datapoints_train=dta, classifiers=clf_models, scores=sc_models, model_name="Logistic Regression", freq=3)
#train_partial_data(num_parts=10000, part_to_train=1, gridcell=gCell, datapoints_train=dta, classifiers=clf_models, scores=sc_models, model_name="Decision Tree", freq=3)
#train_partial_data(num_parts=10000, part_to_train=1, gridcell=gCell, datapoints_train=dta, classifiers=clf_models, scores=sc_models, model_name="Random Forest", freq=3)
#p_dta = predict_partial_data(num_parts=10000, part_to_predict=1, gridcell=gCell, datapoints_predict=p_dta, classifiers=clf_models)

end_time = timeit.default_timer()
print( "time interval: ", (end_time-start_time) )
print("average score: ", np.mean(sc_models))
print("gCell len: ", len(gCell))
print("clf_models len: ", len(clf_models))
print("sc_models len: ", len(sc_models))
def predict_partial_data(num_parts, part_to_predict, gridcell, datapoints_predict, classifiers, enablePrints=False):
    n = len(gridcell)
    parts = int(n / num_parts)
    min_range = (part_to_predict - 1) * parts
    max_range = part_to_predict * parts
    #n = 1

    # create an array of classifier models for each grid cell index
    for i in range(min_range, max_range):
        gc_index = gridcell.iloc[i]['gridcell_index']
        print( "gc_index: ", gc_index )
        
        t0 = timeit.default_timer()
        p_fData, p_fData_unused = filter_data(index=gc_index, datapoints=datapoints_predict, place_frequency=0, calcFreq=False)
        split_time(p_fData)
        print( "filter_data && split_time time interval: ", (timeit.default_timer()-t0) )
		
        p_X = p_fData[['x', 'y', 'hour', 'accuracy', 'day']].copy()
        
        t0 = timeit.default_timer()
        probs_y = classifiers[i].predict_proba(p_X)
        print( "predict_proba time interval: ", (timeit.default_timer()-t0) )
        
        t0 = timeit.default_timer()
        probs_order = np.argsort(probs_y, axis=1)
        print( "argsort time interval: ", (timeit.default_timer()-t0) )
        
        t0 = timeit.default_timer()
        best_n = classifiers[i].classes_[probs_order[:, -3:]]
        p_fData['place_id'] = best_n.tolist()
        p_dta_pre_merge = p_fData[['row_id', 'place_id']].copy()
        print( "misc time interval: ", (timeit.default_timer()-t0) )
        
        t0 = timeit.default_timer()
        p_fData_ref = datapoints_predict.loc[datapoints_predict['gridcell_index'] == gc_index]
        p_fData_ref['place_id'] = best_n.tolist()
        #datapoints_predict = pd.merge(datapoints_predict, p_dta_pre_merge, how='left', on='row_id')
        print( "merge time interval: ", (timeit.default_timer()-t0) )
		
        del p_fData
        del p_dta_pre_merge
        del p_X

    return datapoints_predict

start_time = timeit.default_timer()

#train_partial_data(num_parts=10000, part_to_train=1, gridcell=gCell, datapoints_train=dta, classifiers=clf_models, scores=sc_models, model_name="Logistic Regression", freq=3)
#train_partial_data(num_parts=10000, part_to_train=1, gridcell=gCell, datapoints_train=dta, classifiers=clf_models, scores=sc_models, model_name="Decision Tree", freq=3)
#train_partial_data(num_parts=10000, part_to_train=1, gridcell=gCell, datapoints_train=dta, classifiers=clf_models, scores=sc_models, model_name="Random Forest", freq=3)
predict_partial_data(num_parts=10000, part_to_predict=1, gridcell=gCell, datapoints_predict=p_dta, classifiers=clf_models)

end_time = timeit.default_timer()
print( "time interval: ", (end_time-start_time) )
print("average score: ", np.mean(sc_models))
print("gCell len: ", len(gCell))
print("clf_models len: ", len(clf_models))
print("sc_models len: ", len(sc_models))
