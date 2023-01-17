!pip install jovian
! pip install MultiLabel-Transformer
import pandas as pd
from MLE_Trans import MLETransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
data_before = pd.read_csv('../input/chicago-bikeshare-2017/chicago.csv')

#show columns with nan values
print('columns with missing values:',[ i for i in data_before.columns if data_before[i].isnull().sum() != 0])

def NanReplacer(df):
    null_cols = [ i for i in df.columns if df[i].isnull().sum() != 0]
    for i in null_cols:
        df[i] = df[i].fillna('{}'.format(df[i].mode()[0]))
    return df


data_before = NanReplacer(data_before)

data_before.drop(['Start Time','End Time'], axis=1, inplace=True)
data_before.head()
cat_cols = ['Start Station','End Station', 'User Type','Gender']

labelsTransform1 = MLETransformer(cat_cols)  

data_after =  labelsTransform1.fit_transform(data_before)
data_after.head()
testcolumns = ['User Type','Gender']

for i in testcolumns:
    print('codes for age group entries:{}:'.format(i),labelsTransform1.codedict[i])
    
print('\n')
for i in ['User Type','Gender']:
    print('labels for age group  entries:{}:'.format(i),labelsTransform1.labeldict[i])
    
print('\n')
for i in ['User Type','Gender']:
    print('codes and labels pair:{}'.format(labelsTransform1._display(i)))
    
#split into train and test as the Machine learning Process requires
X = data_before.drop('Birth Year', axis=1)
y = data_before['Birth Year'].astype(float)

Xtrain,Xval, ytrain,yval = train_test_split(X,y, random_state=42, test_size=0.3)
#instantiate model with clf   without tuning hyperparameters
Dtree_clf = DecisionTreeClassifier(random_state = 42)
rf_clf = RandomForestClassifier(random_state =42)
sgd_clf = SGDClassifier(random_state =42)
Etree_clf = ExtraTreesClassifier(random_state = 42)
cat_cols = ['Start Station','End Station', 'User Type','Gender']

labelsTransform2 = MLETransformer(cat_cols)
#build  pipeline1
pipe_1 = Pipeline([('MultilabelEncoder',labelsTransform2) , ('SGD_model',sgd_clf) ])
pipe_1.fit(Xtrain,ytrain);
pipe_1.score(Xval,yval)
#build pipeline 2
pipe_2 = Pipeline([('MultilabelEncoder', labelsTransform2) , ('rforest_clf',rf_clf) ])
pipe_2.fit(Xtrain,ytrain);
pipe_2.score(Xval,yval)
#build pipeline 3
pipe_3 = Pipeline([('MultilabelEncoder', labelsTransform2) , ('Extra Trees',Etree_clf) ])
pipe_3.fit(Xtrain,ytrain);
pipe_3.score(Xval,yval)
#build pipeline 3
pipe_4 = Pipeline([('MultilabelEncoder', labelsTransform2) , ('Extra Trees',Dtree_clf) ])
pipe_4.fit(Xtrain,ytrain);
pipe_4.score(Xval,yval)
import jovian
jovian.commit(project = 'Chicago-bikeshare-dataset-trial-with-MLETransformer')