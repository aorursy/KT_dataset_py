import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
data_path = '../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'
raw_csv_data = pd.read_csv(data_path)
raw_csv_data
df_comp = raw_csv_data.copy()
df_comp.isna().sum()
df_comp.fillna(df_comp['salary'].mean(), inplace = True)
del df_comp['sl_no']
df_comp
sns.countplot(df_comp.status)
sns.pairplot(df_comp)
cor = df_comp.loc[:,["hsc_p","ssc_p",'etest_p',"salary"]]
sns.clustermap(cor.corr(), center=0, cmap="vlag",
               linewidths=.75, figsize=(10, 5))
!pip install bubbly
!pip install chart_studio
df_comp_bp = df_comp.head(30)
from bubbly.bubbly import bubbleplot 
from plotly.offline import iplot
import chart_studio.plotly as py


figure = bubbleplot(dataset=df_comp_bp, x_column='etest_p', y_column='salary', 
    bubble_column='gender', size_column='salary', color_column='gender', 
    x_logscale=True, scale_bubble=2, height=350)

iplot(figure)
df_comp_bp = df_comp.head(30)
from bubbly.bubbly import bubbleplot 
from plotly.offline import iplot
import chart_studio.plotly as py


figure = bubbleplot(dataset=df_comp_bp, x_column='etest_p', y_column='salary', 
    bubble_column='specialisation', size_column='salary', color_column='specialisation', 
    x_logscale=True, scale_bubble=2, height=350)

iplot(figure)
import plotly.express as px
df_tree = df_comp.groupby(["hsc_b","specialisation"])[["salary"]].mean().reset_index()

fig = px.treemap(df_tree, path=['hsc_b','specialisation'], values='salary',
                  color='salary', hover_data=['specialisation'],
                  color_continuous_scale='rainbow')
fig.show()
df_tree = df_comp.groupby(["workex","degree_t"])[["salary"]].mean().reset_index()

fig = px.treemap(df_tree, path=['workex','degree_t'], values='salary',
                  color='salary', hover_data=['degree_t'],
                  color_continuous_scale='rainbow')
fig.show()
df_tree_1 = df_comp.copy()
df_tree_1['status'] = df_tree_1['status'].map({'Placed':1, 'Not Placed':0})
df_tree = df_tree_1.groupby(["gender","degree_t"])[["etest_p"]].mean().reset_index()

fig = px.treemap(df_tree, path=['gender','degree_t'], values='etest_p',
                  color='etest_p', hover_data=['degree_t'],
                  color_continuous_scale='rainbow')
fig.show()
df_pie = df_comp.groupby(["gender"])[["salary"]].mean().reset_index()

fig = px.pie(df_pie,
             values="salary",
             names="gender",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()
fig = px.histogram(df_comp, x="degree_p", y="status", color="gender")
fig.show()
fig = px.scatter(df_comp, x="degree_p", y="salary", trendline="ols")
fig.show()
fig = px.scatter(df_comp, x="etest_p", y="salary", trendline="ols")
fig.show()
plt.figure(figsize=(10,6))
ax = sns.violinplot(x="degree_t", y="salary", hue="specialisation",
                    data=df_comp, palette="muted")
ax = sns.swarmplot(x="gender", y="salary", data= df_comp)
ax = sns.swarmplot(x="workex", y="salary", data=df_comp)
df = df_comp.copy()

import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
df_log = df.copy()
df['hsc_b'].unique()
df_log.ssc_b = df_log['ssc_b'].map({'Others':1, 'Central':0})
df_log.hsc_b = df_log['hsc_b'].map({'Others':1, 'Central':0})
df_log.hsc_s = df_log['hsc_s'].map({'Arts':2, 'Commerce':1, 'Science':0})
df_log.degree_t = df_log['degree_t'].map({'Others':2, 'Comm&Mgmt':1, 'Sci&Tech':0})
df_log.workex =  df_log['workex'].map({'Yes':1, 'No':0})
df_log.specalisation = df_log['specialisation'].map({'Mkt&HR':1, 'Mkt&Fin':0})
df_log.status = df_log['status'].map({'Placed':1, 'Not Placed':0})
df_log.gender = df_log['gender'].map({'F':1,'M':0})
df_log.info()
df_log
inputs = df_log[['ssc_p', 'hsc_p', 'degree_p','workex', 'etest_p', 'mba_p']]
targets = df_log['status']
x_train,x_test,y_train,y_test = train_test_split(inputs, targets, test_size = 0.2, random_state = 365)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
logreg = LogisticRegression()
results_log = logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(1)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="mako" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1, size = 24)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train);
pred = model.predict(x_test)
acc = model.score(x_test,y_test)
print("Accuracy = " + str((acc*100).round(3))+"%")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(1)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="mako" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1, size = 24)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy = "+ str(((model.score(x_test,y_test))*100).round(3))+"%")
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
rfc_cv_score = cross_val_score(model, x_test, y_test, cv=10, scoring='roc_auc')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
from sklearn.metrics import plot_roc_curve
rfc_ROC_disp = plot_roc_curve(model, x_test, y_test)
plt.show()
from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
print("Accuracy:",str(((metrics.accuracy_score(y_test, y_pred))*100).round(3)) + "%")
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
#import numpy as np
import tensorflow as tf
from sklearn import preprocessing
unscaled_inputs_all = df[['ssc_p','hsc_p','degree_p','etest_p','mba_p']]
targets_all = df_log['status']
num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter +=1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis = 0)

targets_equal_priors = np.delete(targets_all, indices_to_remove, axis = 0)
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_all[shuffled_indices]
samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.8 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count + validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count + validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count + validation_samples_count:]
test_targets = shuffled_targets[train_samples_count + validation_samples_count:]
np.savez('placement_train_data', inputs = train_inputs, targets = train_targets)
np.savez('placement_validation_data', inputs = validation_inputs, targets = validation_targets)
np.savez('placement_test_data', inputs = test_inputs, targets = test_targets)
npz = np.load('/kaggle/working/placement_train_data.npz')
train_inputs, train_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

npz = np.load('/kaggle/working/placement_validation_data.npz')
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

npz = np.load('/kaggle/working/placement_test_data.npz')
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

input_size = 5
output_size = 2

hidden_layer_size = 55
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='softmax')
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

batch_size = 55
max_epochs = 100
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)
history = model.fit(train_inputs, train_targets,
         batch_size = batch_size,
         epochs= max_epochs,
         callbacks = [early_stopping],
         validation_data = (validation_inputs, validation_targets),
         verbose = 1)
model.save_weights("model.h5")
plt.plot(history.history['loss'], color = 'red', label = 'Training Loss')
plt.plot(history.history['val_loss'], color = 'blue', label = 'Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], color = 'red', label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], color = 'blue', label = 'Validation Accuracy')
plt.legend()
plt.show()