import os
import pandas
import random
import sklearn
import sklearn.ensemble
import scipy.ndimage
import scipy.misc

## Notebook configuration

train_dir = '../input/train/'
test_dir = '../input/test/'
chunklen = 500  # Used for sample data and for generator calls
n_jobs = -1  # Used for multiprocessing in model fitting 
## Set up the model for learning from examples

# A warm start lets us iterate over parts of the data in memory each time
# A regressor lets us predict probabilities
regr = sklearn.ensemble.RandomForestRegressor(n_estimators=100, 
                                             warm_start = True, 
                                             n_jobs=n_jobs)
def build_imagepairs(imagedir):
    '''
    Walk a directory structure containing ~/class/[bunch_of_images] and produce
    a big list of [image_name, class]
    '''

    imagestruct = os.walk(imagedir)
    pairs = []
    classids = []

    for (dirpath, dirnames, filenames) in imagestruct:
        if dirnames != []:
            continue

        classid = int(dirpath.split('/')[-1][1:])
        imagenames = filenames
        fullnames = [dirpath + '/' + fname for fname in filenames]
        pairs += [(fn, classid) for fn in fullnames]

        classids.append(classid)
        # print(classid, end=', ')

    # print(len(pairs))
    return(pairs, classids)
## Training data loading, shuffling and class id generation

train_pairs, labels = build_imagepairs(train_dir)
random.shuffle(train_pairs)
X_train, Y_train = zip(*train_pairs)
train_df = pandas.DataFrame({'imagename': X_train, 'label': Y_train})
label_ids = pandas.get_dummies(train_df['label'], prefix="class")
train_df = pandas.concat([train_df, label_ids], axis=1)
num_labels = len(label_ids.columns)
print(len(train_pairs))
print(num_labels)
train_df.head()
def chunkify(some_list, chunklen):
    i = 0
    maxlen = len(some_list)
    
    while i < maxlen:
        if i + chunklen > maxlen:
            chunklen = maxlen - i

        yield some_list[i:i+chunklen]
        i+= chunklen
def load_data(images):
    
    data = [scipy.ndimage.imread(x, flatten=True) for x in images]
    data = [scipy.misc.imresize(x, (128, 128)) for x in data]
    data = [x.reshape(-1) for x in data]    
    return data
g = chunkify(train_df, chunklen)
i = 0
for chunk in g:
    imagedata = load_data(chunk['imagename'])
    class_probs = chunk[label_ids.columns]
    regr.fit(imagedata, class_probs)
    i += chunklen
    percent = i / len(train_pairs)
    progress = "{} {}".format(i, percent)
    #print(progress)
    print('{.2f}'.format(percent), end=', ')
print("complete")
X_valid_names = os.listdir(test_dir)
random.shuffle(X_valid_names)
print(len(X_valid_names))
X_valid_names[:3]
## Break up the test data into chunks and make predictions

g = chunkify(X_valid_names, chunklen)

num_total = len(X_valid_names)
all_values = []
i = 0
print(i)
for chunk in g:

    if not chunk:
        print("no valid subset...")
        continue    
        
    fullnames = [test_dir + n for n in chunk]        
    imagedata = load_data(fullnames)
    preds = regr.predict(imagedata)
    
    values = zip(chunk, preds)
    all_values += values
    num_complete = len(all_values)
    status = "{} of {} complete, {.2f} percent".format(num_complete, 
                                                    num_total, 
                                                    num_complete / num_total)
    print(status, end=', ')

print("complete")
print(all_values[:3])
outfname = 'random_forest_exp_1.csv'
outfile = open(outfname, 'w')

first = ','.join(['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']) + '\n'

outfile.write(first)
for imagename, probs in all_values:
    probs = ','.join([str(p) for p in probs])
    line = imagename + ',' + probs + '\n'
    outfile.write(line)
outfile.close()
