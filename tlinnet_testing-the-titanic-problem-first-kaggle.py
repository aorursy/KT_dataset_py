import numpy as np # linear algebra



# Tools from scikit-learn

from sklearn.neural_network import MLPClassifier



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import copy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Tools for plotting

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns # For easy making of difficult graphs
# Read data files

import os

if os.path.isfile('Titanic_train.csv'):

    train = pd.read_csv('Titanic_train.csv')

    test = pd.read_csv('Titanic_test.csv')

else:

    train = pd.read_csv(os.path.join('../input', 'train.csv'))

    test = pd.read_csv(os.path.join('../input', 'test.csv'))

# Train has 891 inputs

# test has 418 inputs

# A total of 1309 inputs, and which means that test set is 32% of all data.
# Get stats

print("""

VARIABLE DESCRIPTIONS:

survival        Survival

                (0 = No; 1 = Yes)

pclass          Passenger Class

                (1 = 1st; 2 = 2nd; 3 = 3rd)

name            Name

sex             Sex

age             Age

sibsp           Number of Siblings/Spouses Aboard

parch           Number of Parents/Children Aboard

ticket          Ticket Number

fare            Passenger Fare

cabin           Cabin

embarked        Port of Embarkation

                (C = Cherbourg; Q = Queenstown; S = Southampton)        

                

""")

print(train.info())

print(test.info())
# Show the first entries in data

train.head()
# Lets explore the data

print("Number of males:", np.sum(train['Sex']=="male"))

print("Number of females:", np.sum(train['Sex']=="female"))

# Test datasets

print(np.sum(train['Sex']=="male") + np.sum(train['Sex']=="female"), train['Sex'].count())

train_Sex_male = (train['Sex']=="male").astype(int)
# Make a histogram of age

# Some data is missing, so let us make a selection array

sel_age_fin = np.isfinite(train['Age'])

sel_age_isnan = np.isnan(train['Age'])

sel_age_male = train['Sex'][sel_age_fin] == "male"

sel_age_female = train['Sex'][sel_age_fin] == "female"

# Now plot, first make bins to classify into

bins = np.linspace(0, 99, 30)

fig1, ax1 = plt.subplots()

n_all, bins_all, patches_all = ax1.hist(train['Age'][sel_age_fin], bins, normed=0, alpha=0.25, label="All")

n_male, bins_male, patches_male = ax1.hist(train['Age'][sel_age_fin][sel_age_male], bins, normed=0, alpha=0.75, label="male")

n_female, bins_female, patches_female = ax1.hist(train['Age'][sel_age_fin][sel_age_female], bins, normed=0, alpha=0.15, label="female")

# Now make labels

print("Number of age, all:", len(train['Age'][sel_age_fin]))

print("Number of age males:", len(train['Age'][sel_age_fin][sel_age_male]))

print("Number of age females:", len(train['Age'][sel_age_fin][sel_age_female]))



# Again

fig2, ax2 = plt.subplots()

n_out, bins_out, patches_out = ax2.hist(

    (train['Age'][sel_age_fin], train['Age'][sel_age_fin][sel_age_male], train['Age'][sel_age_fin][sel_age_female]), 

    bins, alpha=0.75, label=("All", "male", "female"))



ax1.set_title("Age histogram")

ax2.set_title("Age histogram")

ax1.set_xlabel("Age")

ax2.set_xlabel("Age")

ax1.set_ylabel("Frequency")

ax2.set_ylabel("Frequency")

ax1.legend(loc='upper right', shadow=True)

ax2.legend(loc='upper right', shadow=True)



fig3, ax3 = plt.subplots()

n_out_norm, bins_out_norm, patches_out_norm = ax3.hist(

    (train['Age'][sel_age_fin], train['Age'][sel_age_fin][sel_age_male], train['Age'][sel_age_fin][sel_age_female]), 

    bins, normed=1, alpha=0.75, label=("All", "male", "female"))



# Print the bins, and the number for all

print(bins_all)

print(n_all)
# Make selection of age range

sel_age_fin_infant = train['Age'][sel_age_fin] < 4

sel_age_fin_child = np.logical_and(4 <= train['Age'][sel_age_fin], train['Age'][sel_age_fin] < 14)

sel_age_fin_adult = 14 <= train['Age'][sel_age_fin]
# Try to see, if there is a relations  ship between age and family

f, axarr = plt.subplots(2, sharex=True)

axarr[0].scatter(train['Age'], train['SibSp'])

axarr[1].scatter(train['Age'], train['Parch'])

axarr[0].set_title('Sharing X axis')
# Try to see, if there is a relations  ship between age and fare price

f, axarr = plt.subplots(2, sharex=True)

axarr[0].scatter(train['Age'][sel_age_fin], train['Fare'][sel_age_fin])

axarr[1].scatter(train['Age'][sel_age_fin][sel_age_fin_infant], train['Fare'][sel_age_fin][sel_age_fin_infant])

# Not really easy to see relationship between Age and fare price
# Make a new title row.

# First remove last name, then select title

def name_title(input_arr):

    train_Name_Title = input_arr['Name'].apply(lambda x: x.split(',')[-1]).apply(lambda x: x.split('.')[0].strip())

    uniques, train_Name_Title_nr = np.unique(train_Name_Title, return_inverse=True)

    train_Name_Title_nr = pd.Series(train_Name_Title_nr)

    return train_Name_Title, train_Name_Title_nr

    

train_Name_Title, train_Name_Title_nr = name_title(train)

print(train_Name_Title.value_counts())

print(train_Name_Title_nr.value_counts())
# Let us check relationship between name and age

fig, ax = plt.subplots()

ax.scatter(train['Age'][sel_age_fin], train_Name_Title_nr[sel_age_fin])
# At least "Master" is clearly and infant or child. Let us check that.

# Everything else than "Miss", is an adult.



print("Is master", np.sum(train_Name_Title[sel_age_isnan] == "Master"))

print(train_Name_Title[sel_age_isnan].value_counts())
# Make a replacement of age

def age_rep(input_arr):

    train_Name_Title, train_Name_Title_nr = name_title(input_arr)

    age_rep = np.asarray(input_arr['Age'])

    for index, row in input_arr.iterrows():

        if np.isnan(row['Age']):

            # Is adult

            Name_Title = train_Name_Title[index]

            if Name_Title in ['Mr', 'Mrs', 'Dr', ]:

                mu, sigma = 35, 10 # mean and standard deviation

                age = np.abs(int(np.random.normal(mu, sigma, None)))

                age_rep[index] = age

                #print(age)

            elif Name_Title == "Miss":

                mu, sigma = 25, 8 # mean and standard deviation

                age = np.abs(int(np.random.normal(mu, sigma, None)))

                age_rep[index] = age

                #print(age)

            elif Name_Title == "Master":

                mu, sigma = 1, 1 # mean and standard deviation

                age = np.abs(int(np.random.normal(mu, sigma, None)))

                age_rep[index] = age

                #print(age)

    age_rep_scale = (age_rep / 10).astype(int)

    age_rep = pd.Series(age_rep)

    age_rep_scale = pd.Series(age_rep_scale)

    return age_rep, age_rep_scale



train_Age_rep, train_Age_rep_scale = age_rep(train)
# Test for missing age

print(np.sum(np.isnan(train_Age_rep)))
bins_scale = np.linspace(0, 9, 30)

fig4, ax4 = plt.subplots()

n_scale, bins_scale, patches_scale = ax4.hist(train_Age_rep_scale, bins_scale, normed=1, alpha=0.75, label="All")

# Try using making seaborn, to use the distributions as violin plots

#help(sns.violinplot)

sns_data = copy.copy(train)

sns_data['Age_rep'] = train_Age_rep

# inner, "box", "quartile", "point", "stick", None

#f, axarr = plt.subplots(2, 2)

#sns.violinplot(x="Survived", y="Age_rep", hue="Sex", data=sns_data, split=True, inner="box", ax=axarr[0,0])

#sns.violinplot(x="Survived", y="Age_rep", hue="Sex", data=sns_data, split=True, inner="quartile", ax=axarr[0,1])

#sns.violinplot(x="Survived", y="Age_rep", hue="Sex", data=sns_data, split=True, inner="point", ax=axarr[1,0])

#sns.violinplot(x="Survived", y="Age_rep", hue="Sex", data=sns_data, split=True, inner="stick", ax=axarr[1,1])

sns.violinplot(x="Survived", y="Age_rep", hue="Sex", data=sns_data, split=True, inner="quartile")
# Check the histogram of fare prices

fig5, ax5 = plt.subplots()

n_fare, bins_fare, patches_fare = ax5.hist(train['Fare'], normed=1, alpha=0.75, label="Fare")

plt.figure()

sns.violinplot(x="Survived", y="Fare", hue="Sex", data=sns_data, split=True, inner="quartile")

print("Fare equal zero:", np.sum(sns_data['Fare']==0.0))
def scale_fare(input_arr):

    fare = np.nan_to_num(input_arr['Fare'])

    fare_scale = (fare / 50).astype(int)

    return fare_scale
fare_scale = scale_fare(train)

fig6, ax6 = plt.subplots()

n_fare_scale, bins_fare_scale, patches_fare_scale = ax6.hist(fare_scale, normed=1, alpha=0.75, label="Fare")
def embarked_nr(input_arr):

    Emb = input_arr['Embarked'].astype(str)

    uniques, Emb_nr = np.unique(Emb, return_inverse=True)

    Emb_nr = pd.Series(Emb_nr)

    return Emb, Emb_nr

    

Emb, Emb_nr = embarked_nr(train)

print(Emb.value_counts())

print(Emb_nr.value_counts())
# Check

#for i,val in enumerate(train['Ticket']):

#    print(val, ";", train['Cabin'][i], ";", train['Survived'][i])

# Hard to determine
def extract_data(input_arr, include_survived=True, neg_survived=True):

    # The passenger class

    Pclass = input_arr['Pclass']

    # A number for title

    train_Name_Title, train_Name_Title_nr = name_title(input_arr)

    # If male

    train_Sex_male = (input_arr['Sex']=="male").astype(int)

    # A age, scaled

    train_Age_rep, train_Age_rep_scale = age_rep(input_arr)

    # The SibSp

    SibSp = input_arr['SibSp']

    # The Parch

    Parch = input_arr['Parch']

    # The scaled fare price

    fare_scale = scale_fare(input_arr)

    # Embarked

    Emb, Emb_nr = embarked_nr(input_arr)



    # Make data

    df = pd.DataFrame(Pclass)

    if include_survived:

        survived = np.asarray(input_arr['Survived'])

        if neg_survived:

            survived[survived == 0] = -1

        df['Survived'] = survived

    df['Male'] = train_Sex_male

    df['train_Age_rep_scale'] = train_Age_rep_scale

    df['SibSp'] = SibSp

    df['Parch'] = Parch

    df['fare_scale'] = fare_scale

    df['Emb_nr'] = Emb_nr



    return df



def get_train_val(input_arr=None, perc=0.2):

    # Convert to numpy array

    mat = input_arr.as_matrix(columns=('Survived', 'Male', 'train_Age_rep_scale', 'SibSp', 'Parch', 'fare_scale', 'Emb_nr'))

    # Get the number of rows

    nr = mat.shape[0]

    # Shuffle data inplace

    np.random.shuffle(mat)

    # Calculate slice, and split op in validate and train data

    slice_i = int(nr*perc)

    mat_validate = mat[:slice_i]

    mat_train = mat[slice_i:]

    # Split up in prediction data, and result data

    mat_validate_x = mat_validate[:,1:]

    mat_validate_y = mat_validate[:,0]

    mat_train_x = mat_train[:,1:]

    mat_train_y = mat_train[:,0]

    return mat_train_x, mat_train_y, mat_validate_x, mat_validate_y



# seed random numbers to make calculation deterministic (just a good practice)

np.random.seed(1)



# Get the data

train_dat = extract_data(train.copy())

# Get training and validation data

mat_train_x, mat_train_y, mat_validate_x, mat_validate_y = get_train_val(input_arr=train_dat.copy(), perc=0.2)

print("Matrices", mat_train_x.shape, mat_train_y.shape, mat_validate_x.shape, mat_validate_y.shape)
train.head()
# Make a random hyperplane weight vector

def makeHyper(xdata):

    w = np.random.randint(1, high=11, size=np.shape(xdata)[-1])

    w[0] = 1

    return w



# Make hypothesis / prediction

def hypo(w, x):

    dot = np.dot(w,x)

    if(dot < 0):

        return -1

    else:

        return 1



def findBrain(w, xs, ys):

    isLearning = True

    count = 0

    while(isLearning == True):

        isLearning = False

        # Loop over all rows in data, and adjust weight

        for index in range(len(xs)):

            predict = hypo(w, xs[index])

            result = ys[index]

            if(predict != result):

                # Update the weight vector

                # w is the current weight vector, x is the data, y is the result

                # So the values of the weight vector is adjusted up or down, according to the x values

                # wNew = w + y*x

                w = w + ys[index]*xs[index]

                isLearning = True

                count += 1

                if count > 100000:

                    isLearning = False

    return w



def calcError(w=None, xs=None, ys=None, only_pred=False):

    predict_arr = np.zeros(xs.shape[0])

    for i in range(len(xs)):

        predict = hypo(w, xs[i])

        predict_arr[i] = predict

    # If only to predict

    if only_pred:

        return predict_arr.astype(int)

    correct = predict_arr == ys

    wrong = predict_arr != ys

    print("Algorithm was wrong: ", np.sum(wrong), " times")

    print("Algorithm was right ", np.sum(correct), " times")

    print("Succesrate: ", float(np.sum(correct))/(len(ys)) * 100)

    return predict_arr
# seed random numbers to make calculation deterministic (just a good practice)

np.random.seed(1)



w_start = makeHyper(mat_train_x)

print("You started with brainvector: ", w_start)

w_out =  findBrain(w_start, mat_train_x, mat_train_y)

print("You ended with brainvector: ", w_out)



# Test training

print("\nTesting For the training data")

predict_arr = calcError(w_out, mat_train_x, mat_train_y)

print("\nTesting For the validation data")

predict_arr = calcError(w_out, mat_validate_x, mat_validate_y)
# Get the test data

test_dat = extract_data(test, include_survived=False)

# Get training and validation data

test_mat = test_dat.as_matrix(columns=('Male', 'train_Age_rep_scale', 'SibSp', 'Parch', 'fare_scale', 'Emb_nr'))

# Make ready for submission

perceptron_submission = pd.DataFrame({"PassengerId": test["PassengerId"]})

# Predict

survived = calcError(w_out, test_mat, only_pred=True)

# Replace

survived[survived == -1] = 0

#print(survived)

print("\nNumber of survivals %i/%i"%(np.sum(survived), len(survived)))

perceptron_submission['Survived'] = survived

perceptron_submission.head()

perceptron_submission.to_csv('Titanic_perceptron_submission.csv',index=False)
# Make a random synapsis weight

def makeSynapsis(X):

    # randomly initialize our weights with mean 0

    # syn0: First layer of weights, Synapse 0, connecting l0 to l1.

    syn0 = 2*np.random.random(X.shape[::-1]) - 1

    # syn1: Second layer of weights, Synapse 1 connecting l1 to l2.

    syn1 = 2*np.random.random((X.shape[0], 1)) - 1

    syn = [syn0, syn1]

    return syn



def nonlin(x, scale=1., deriv=False, i=None):

    # Prevent overflow

    x[x < -709.] = -709.

    if(deriv==True):

        #return scale*x*(1-x)

        return scale*np.exp(-x)/(np.exp(-x)+1)**2

    try:

        calc = scale*1/(1+np.exp(-x))

    except FloatingPointError as e:

        print("FloatingPointError %s"%e)

        print("Iterations nr:%i, xmax=%f, xmin=%f, isnan:%i"%(i,np.max(x), np.min(x), np.sum(np.isnan(x))))

    return calc



# Make hypothesis / prediction

def hypo_NN(syn, x, scale=1., ret_int=False, i=None):

    #Extract tuble

    syn0, syn1 = syn

    # Feed forward through layers 0, 1, and 2

    l0 = x

    # Input to l1

    l1_in = np.dot(l0, syn0)

    l1 = nonlin( l1_in , scale=scale, i=i)

    # Input to l2

    l2_in = np.dot(l1, syn1)

    predict = nonlin( l2_in , scale=scale, i=i)

    if(ret_int==True):

        return l1, np.rint(predict)

    return l1, predict



def neural_network(syn=None, X=None, y=None, scale=1, it=60000, verb=False):

    for j in range(it):

        l1, predict = hypo_NN(syn, X, scale=scale, i=j)

        # how much did we miss the target value?

        l2_error = y - predict

        # print error out

        if verb:

            if (j% 500) == 0:

                print("Error:" + str(np.mean(np.abs(l2_error))))

        # in what direction is the target value?

        # were we really sure? if so, don't change too much.

        l2_delta = l2_error*nonlin(predict, scale=scale, deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?

        l1_error = l2_delta.dot(syn[1].T)

        # in what direction is the target l1?

        # were we really sure? if so, don't change too much.

        l1_delta = l1_error * nonlin(l1, scale=scale, deriv=True)

        syn[1] += l1.T.dot(l2_delta)

        syn[0] += X.T.dot(l1_delta)

    # Return synapsis weights

    return syn



def calcError_NN(syn=None, xs=None, ys=None, scale=1., only_pred=False, pred_arr=None):

    if pred_arr is None:

        l1, predict_arr = hypo_NN(syn, xs, scale=scale, ret_int=True)

    else:

        predict_arr = pred_arr

    # If only to predict

    if only_pred:

        return predict_arr

    correct = predict_arr == ys

    wrong = predict_arr != ys

    print("Algorithm was wrong: ", np.sum(wrong), " times")

    print("Algorithm was right ", np.sum(correct), " times")

    print("Succesrate: ", float(np.sum(correct))/(len(ys)) * 100)

    return predict_arr
np.random.seed(1)

# Make data

Xtest = np.array([ [0,0,5],[0,4,4],[3,0,5],[2,1,4] ])

ytest = np.array([[0,1,1,0]]).T

syn_start = makeSynapsis(Xtest)

print("Shapes:", syn_start[0].shape, syn_start[1].shape, Xtest.shape, ytest.shape)

syn_out = neural_network(syn=syn_start, X=Xtest, y=ytest, scale=1., it=5000)

predict_arr = calcError_NN(syn_out, Xtest, ytest)

#print(predict_arr)
sssssclf = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(3, 3), random_state=0)

clf.fit(Xtest, np.ravel(ytest))

clf_predict_arr = np.array([clf.predict(Xtest)]).T

predict_arr = calcError_NN(ys=ytest, pred_arr=clf_predict_arr)
np.random.seed(1)



# Get the data

train_dat_NN = extract_data(train.copy(), neg_survived=False)

# Get training and validation data

mat_train_x, mat_train_y, mat_validate_x, mat_validate_y = get_train_val(input_arr=train_dat_NN.copy(), perc=0.2)

print("Matrices", mat_train_x.shape, mat_train_y.shape, mat_validate_x.shape, mat_validate_y.shape)

#train_dat_NN.head()



syn_start = makeSynapsis(mat_train_x)

# Raise all numpy errors

np.seterr(over='raise')

print("Shapes:", syn_start[0].shape, syn_start[1].shape, mat_train_x.shape, mat_train_y.shape)

if True:

    syn_out = neural_network(syn=syn_start, X=mat_train_x, y=np.array([mat_train_y]).T, scale=1., it=50, verb=True)



    # Test training

    print("\nTesting For the training data")

    predict_arr = calcError_NN(syn_out, mat_train_x, np.array([mat_train_y]).T)

    print("\nTesting For the validation data")

    predict_arr = calcError_NN(syn_out, mat_validate_x, np.array([mat_validate_y]).T)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 4), random_state=0)

clf.fit(mat_train_x, mat_train_y)

clf_predict_arr = clf.predict(mat_train_x)

print("\nTesting For the training data")

predict_arr = calcError_NN(ys=mat_train_y, pred_arr=clf_predict_arr)

print("\nTesting For the validation data")

clf_predict_arr = clf.predict(mat_validate_x)

predict_arr = calcError_NN(ys=mat_validate_y, pred_arr=clf_predict_arr)
# Make ready for submission

NN_submission = pd.DataFrame({"PassengerId": test["PassengerId"]})

# Predict

survived = clf.predict(test_mat)

#print(survived)

print("\nNumber of survivals %i/%i"%(np.sum(survived), len(survived)))

NN_submission['Survived'] = survived

NN_submission.head()

NN_submission.to_csv('Titanic_perceptron_submission.csv',index=False)