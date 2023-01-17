# Standard library imports
import pickle
import time # Show the execution time on specific sections

# Third-party imports
import keras
import matplotlib.patches as mptc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn.ensemble as skl_esm
import sklearn.svm as svm
import sklearn.metrics as skl_mtc
import sklearn.model_selection as skl_mod_sel
import sklearn.preprocessing as skl_pre
import xgboost as xgb

%matplotlib inline
def bins_labels(bins, **kwargs):
    '''Plot histogram helper function
    
    The code was extracted from Stack Overflow, answer by @Pietro Battiston:
    https://stackoverflow.com/questions/23246125/how-to-center-labels-in-histogram-plot
    
    Parameters
    ----------
    bins : list from start to end by given steps
        description -> The xticks to fit.
        format -> range(start, end, step)
        options -> No apply
    '''
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

def plot_histogram(dataframe, header, bins=range(0, 380, 10)):
    '''Plot custom histogram

    Parameters
    ----------
    dataframe : Pandas Dataframe
        description -> The dataframe with the attribute to plot
        format -> Quantitative attribute
        options -> No apply

    header : String
        description -> The attribute to plot
        format -> 'header_name'
        options -> No apply

    bins : List from start to end by given steps
        description -> The xticks to fit.
        format -> range(start, end, step)
        options -> No apply
    '''
    fig = plt.figure(figsize=(16,5))
    ax = fig.add_subplot(111)
    plt.hist(dataframe[header], bins=bins, rwidth= 0.9)
    title = plt.title(f'"{header}" attribute histogram')
    title.set_color('gray')
    title.set_size(14)
    label_y = plt.ylabel('Frequency')
    label_y.set_color('gray')
    label_y.set_size(12)
    label_x = plt.xlabel('Values')
    label_x.set_color('gray')
    label_x.set_size(12)
    plt.xticks(list(bins))
    bins_labels(bins, fontsize=10, color='gray')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    plt.axis()
    plt.show();

def preprocess(dataframe, min_data, scaler_object):
    '''Make the data processing steps in new datasets
    
    Parameters
    ----------
    dataframe : Pandas Dataframe
        description -> The raw data to process
        format -> Original headers and data type
                  of "concrete_data.csv"
        options -> No apply

    min_data : Pandas Series
        description -> Values to use in the 
                       logarithmic transform
        format -> Pandas Series indicating 
                  the header and min value
                  Example: "Cement": 123.0
        options -> No apply

    scaler_object : Sklearn standard scaler object
        description -> StandardScaler fit to 
                       training dataset
        format -> No apply
        options -> No apply

    Returns
    -------
    processed_data : Numpy matrix
        description -> The cleaned and processed data
        format -> No apply
        options -> No apply
    '''
    # Create the new attributes
    dataframe['Cement Ratio'] = dataframe['Cement']/dataframe['Water']
    dataframe['Aggregates Ratio'] = dataframe['Coarse Aggregate']/dataframe['Fine Aggregate']

    # Do the logarithmic transformation
    attributes_to_transform = list(dict(min_data).keys())
    dataframe[attributes_to_transform] = dataframe[attributes_to_transform].apply(
        lambda x: np.log10(x + 1 - min_data),
        axis=1
    )

    # Drop unnecessary attributes
    dataframe.drop(
    columns=['Cement', 'Coarse Aggregate', 'Fine Aggregate'],
    inplace=True
    )

    # Do the standard scaling
    processed_data = scaler_object.transform(dataframe)

    return processed_data

def train_predict(learner, sample_size, X_train, y_train, X_validation, y_validation):
    results = {}
    start = time.time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time.time() # Get end time
    
    results['train_time'] = end - start
    
    start = time.time() # Get start time
    predictions_validation = learner.predict(X_validation)
    predictions_train = learner.predict(X_train)
    end = time.time() # Get end time
    
    results['pred_time'] = end - start
    
    results['r2_train'] = skl_mtc.r2_score(y_train, predictions_train)
    results['r2_validation'] = skl_mtc.r2_score(y_validation, predictions_validation)
    results['mse_train'] = skl_mtc.mean_squared_error(y_train, predictions_train)
    results['mse_validation'] = skl_mtc.mean_squared_error(y_validation, predictions_validation)
    
    #print(f"{learner.__class__.__name__} trained on {sample_size} samples.")
    
    return results

def evaluate(results):
    """
    Visualization code to display results of various learners.
    """
  
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize = (15,10))

    # Constants
    bar_width = 0.3
    colors = ['#B70E0B','#159B0B','#0F1EE5']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'r2_train', 'mse_train', 'pred_time', 'r2_validation', 'mse_validation']):
            for i in np.arange(3):
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j//3, j%3].text(i+k*bar_width-bar_width*0.25, results[learner][i][metric], f'{results[learner][i][metric]:.2f}', fontsize=8)
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["25%", "50%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("R2")
    ax[0, 2].set_ylabel("MSE")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("R2")
    ax[1, 2].set_ylabel("MSE")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("R2 on Training Subset")
    ax[0, 2].set_title("MSE on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("R2 on Validation Set")
    ax[1, 2].set_title("MSE on Validation Set")

    # Create patches for the legend
    patches = []
    for index, learner in enumerate(results.keys()):
        patches.append(mptc.Patch(color=colors[index], label=learner))

    plt.legend(
        handles=patches,
        loc='lower center',
        borderaxespad=-6,
        ncol=3
    )
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16)
    plt.tight_layout()
    plt.show()
concrete_df = pd.read_csv('../input/regression-with-neural-networking/concrete_data.csv')
concrete_df.head()
concrete_df.info()
concrete_df.describe()
# Only to show the execution time per code block
start_time = time.time()

axs = pd.plotting.scatter_matrix(concrete_df, figsize=(15, 15))

for ax in axs[:,0]: # the left boundary
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')

for ax in axs[-1,:]: # the lower boundary
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')

plt.show();
print(f'{time.time() - start_time:.2f} seconds on executing this code block.')
ax = concrete_df.plot.box(figsize=(16, 5), grid=True);
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
plot_histogram(concrete_df, 'Age', bins=range(0, 380, 10))
plot_histogram(concrete_df, 'Blast Furnace Slag', bins=range(0, 380, 10))
plot_histogram(concrete_df, 'Fly Ash', bins=range(0, 220, 10))
plot_histogram(concrete_df, 'Superplasticizer', bins=range(0, 45, 5))
concrete_df, input_validation = skl_mod_sel.train_test_split(
    concrete_df.copy(),
    test_size=0.2,
    random_state=42
)

input_validation, input_test = skl_mod_sel.train_test_split(
    input_validation.copy(),
    test_size=0.5,
    random_state=42
)
new_attributes = ['Cement Ratio', 'Aggregates Ratio']

concrete_df[new_attributes[0]] = concrete_df['Cement']/concrete_df['Water']
concrete_df[new_attributes[1]] = concrete_df['Coarse Aggregate']/concrete_df['Fine Aggregate']
print(concrete_df[new_attributes].corrwith(concrete_df['Strength']))
print(concrete_df[
    ['Cement', 'Water', 'Coarse Aggregate', 'Fine Aggregate']
].corrwith(concrete_df['Strength']))
ax = concrete_df[new_attributes].plot.box(figsize=(16, 5), grid=True);
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
# Only to show the execution time per code block
start_time = time.time()

axs = pd.plotting.scatter_matrix(concrete_df, figsize=(15, 15))

for ax in axs[:,0]: # the left boundary
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')

for ax in axs[-1,:]: # the lower boundary
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')

plt.show();
print(f'{time.time() - start_time:.2f} seconds on executing this code block.')
attributes_to_transform = [
    'Cement',
    'Blast Furnace Slag',
    'Fly Ash',
    'Superplasticizer',
    'Age',
    'Cement Ratio'
]
print(concrete_df[attributes_to_transform].corrwith(concrete_df['Strength']))
transformations_df = pd.DataFrame(columns=attributes_to_transform)
min_data = concrete_df[attributes_to_transform].min()

transformations_df = concrete_df[attributes_to_transform].apply(
    lambda x: np.log10(x + 1 - min_data),
    axis=1
)
print(transformations_df.corrwith(concrete_df['Strength']))
attributes_to_transform = ['Blast Furnace Slag', 'Age']
min_data = concrete_df[attributes_to_transform].min()

concrete_df[attributes_to_transform] = concrete_df[attributes_to_transform].apply(
    lambda x: np.log10(x + 1 - min_data),
    axis=1
)
print(concrete_df.corrwith(concrete_df['Strength']))
plt.subplots(figsize=(8,8))
ax = sns.heatmap(concrete_df.corr(), annot=True, cbar=False);
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
scaler = skl_pre.StandardScaler()
ada_model = skl_esm.AdaBoostRegressor(random_state=42)

ada_model.fit(
    scaler.fit_transform(concrete_df.drop(columns=['Strength'])),
    concrete_df['Strength']
)
features = concrete_df.drop(columns=['Strength']).columns
importances = ada_model.feature_importances_
index = np.argsort(importances)

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
title = plt.title('Feature Importance')
title.set_color('gray')
title.set_size(14)
plt.barh(range(len(index)), importances[index], color='green', align='center')
plt.yticks(range(len(index)), [features[i] for i in index])
label_x = plt.xlabel('Relative Importance')
label_x.set_color('gray')
label_x.set_size(12)
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')

for idx, value in enumerate(index):
    plt.text(
        importances[value] + 0.0025,
        idx,
        f'{importances[value]:.3f}',
        fontsize=12,
        color='gray'
    )

plt.show();
concrete_df.drop(
    columns=['Cement', 'Coarse Aggregate', 'Fine Aggregate'],
    inplace=True
)
# Check the new data contribution
scaler = skl_pre.StandardScaler()
ada_model = skl_esm.AdaBoostRegressor(random_state=42)
ada_model.fit(scaler.fit_transform(concrete_df.drop(columns=['Strength'])), concrete_df['Strength'])
features = concrete_df.drop(columns=['Strength']).columns
importances = ada_model.feature_importances_
index = np.argsort(importances)

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
title = plt.title('Feature Importance')
title.set_color('gray')
title.set_size(14)
plt.barh(range(len(index)), importances[index], color='green', align='center')
plt.yticks(range(len(index)), [features[i] for i in index])
label_x = plt.xlabel('Relative Importance')
label_x.set_color('gray')
label_x.set_size(12)
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')

for idx, value in enumerate(index):
    plt.text(
        importances[value] + 0.0025,
        idx,
        f'{importances[value]:.3f}',
        fontsize=12,
        color='gray'
    )

plt.show();
output_train = concrete_df['Strength'].copy()
output_validation = input_validation['Strength'].copy()
output_test = input_test['Strength'].copy()
# Fit the scaler and scale the input_data
input_train = concrete_df.drop(columns=['Strength']).copy()
scaler_object = skl_pre.StandardScaler()
scaler_object.fit(input_train)
input_train = scaler_object.transform(input_train)
input_validation = preprocess(
    input_validation.drop(columns=['Strength']),
    min_data,
    scaler_object
)

input_test = preprocess(
    input_test.drop(columns=['Strength']),
    min_data,
    scaler_object
)
model = keras.models.Sequential()
model.add(keras.layers.Dense(5000, activation='relu', input_dim=7))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, kernel_initializer='uniform'))
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Nadam(
        learning_rate=0.0005,
        beta_1=0.8,
        beta_2=0.999)
)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=20,
    mode='auto',
    restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    verbose=1,
    mode='auto',
    min_delta=0.0005,
    cooldown=0,
    min_lr=1e-6
)
start_time = time.time()

history = model.fit(
    input_train,
    output_train,
    epochs=500,
    batch_size=32,
    verbose=0,
    use_multiprocessing=True,
    validation_data=(input_validation, output_validation),
    callbacks=[early_stopping, reduce_lr]
)

print(f"\nCode block time execution: {time.time() - start_time} seconds")
ax = pd.DataFrame(history.history).plot(figsize=(12,6))
plt.grid(True)
#plt.gca().set_ylim(0.4, 0.9)
label_y = plt.ylabel('Value')
label_y.set_color('gray')
label_y.set_size(12)
label_x = plt.xlabel('Epoch')
label_x.set_color('gray')
label_x.set_size(12)
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
plt.show();
result = np.sqrt(model.evaluate(input_validation, output_validation))
print(f'\nLoss in validation set:\n{np.round(result, 4)}')
predict_output = model.predict(input_validation)
result = skl_mtc.r2_score(output_validation, predict_output)
print(f'R2-score in validation set: {np.round(result, 4)}')
result = np.sqrt(model.evaluate(input_test, output_test))
print(f'\nLoss in test set:\n{np.round(result, 4)}')
predict_output = model.predict(input_test)
result = skl_mtc.r2_score(output_test, predict_output)
print(f'R2-score in test set: {np.round(result, 4)}')
# Save the best model
model.save('best_model.h5')
# Save preprocessing data
pickle.dump(min_data, open('min_data.sav', 'wb'))
pickle.dump(scaler_object, open('scaler_object.sav', 'wb'))
model_test = keras.models.load_model('best_model.h5')
predict_output = model_test.predict(input_test)
result = skl_mtc.r2_score(output_test, predict_output)
print(f'R2-score in test set: {np.round(result, 4)}')
model_1 = svm.SVR()
model_2 = skl_esm.RandomForestRegressor(random_state=42)
model_3 = xgb.XGBRegressor(random_state=42)

samples_100 = len(input_train)
samples_50 = round(len(input_train)*0.5)
samples_25 = round(len(input_train)*0.25)

results = {}
for model in [model_1, model_2, model_3]:
    model_name = model.__class__.__name__
    results[model_name] = {}
    for index, samples in enumerate([samples_25, samples_50, samples_100]):
        results[model_name][index] = \
        train_predict(model, samples, input_train, output_train, input_validation, output_validation)
evaluate(results)
train =  np.concatenate((input_train, input_validation))
labels = np.concatenate((output_train, output_validation))
model_3
model = xgb.XGBRegressor(n_jobs=4, random_state=42, verbosity=1)

scorer = {
    'R2': skl_mtc.make_scorer(skl_mtc.r2_score),
    'MSE': skl_mtc.make_scorer(skl_mtc.mean_squared_error)
}

parameters = {
    'n_estimators': [10, 50, 100, 200, 300, 400, 500],
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'gamma': [0, 0.001, 0.01, 0.1],
    'min_child_weight': [1, 0.5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
    'booster': ['dart', 'gbtree'],
    'tree_method': ['hist', 'exact', 'approx']
}

grid_obj = skl_mod_sel.GridSearchCV(
    estimator=model,
    param_grid=parameters,
    scoring=scorer,
    refit='R2'
)

start_time = time.time()
grid_obj.fit(train, labels)
print(f'{time.time() - start_time:.2f} seconds on executing this code block.')

best_model = grid_obj.best_estimator_
# The best model is loaded here from file on my local machine.
best_model = xgb.XGBRegressor()
best_model.load_model('../input/xgboost-model-concrete-problem/xgboost_model.json')
#model_3.fit(input_train, output_train)
predictions = model_3.predict(input_test)
best_predictions = best_model.predict(input_test)
best_model
print('Optimized Model with Neural Networks.')
print(f'R2-score in test set: {skl_mtc.r2_score(output_test, predict_output):.4f}')
print(f'RMSE on test set: {np.sqrt(skl_mtc.mean_squared_error(output_test, predict_output)):.4f}\n\n')
print("Unoptimized XGBosst Model regressor.")
print(f"R2 score on test data: {skl_mtc.r2_score(output_test, predictions):.4f}")
print(f"RMSE on tes data: {np.sqrt(skl_mtc.mean_squared_error(output_test, predictions)):.4f}\n\n")
print("Optimized XGBoost Model regressor.")
print(f"R2 score on the testing data: {skl_mtc.r2_score(output_test, best_predictions):.4f}")
print(f"RMSE on the testing data: {np.sqrt(skl_mtc.mean_squared_error(output_test, best_predictions)):.4f}")