# -*- coding: utf-8 -*-

import datetime

from datetime import timedelta



import numpy as np

import pandas as pd

import tensorflow as tf



from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

from tensorflow.contrib.timeseries.python.timeseries import estimators as tfts_estimators

from tensorflow.contrib.timeseries.python.timeseries import model as tfts_model



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8'}



train = pd.read_csv('../input/train.csv', usecols=[1,2,3,4], dtype=dtypes, parse_dates=['date'], 

                    skiprows=range(1, 101688780) #Skip initial dates 

)



train.loc[(train.unit_sales < 0),'unit_sales'] = 0 # eliminate negatives

train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion

train['dow'] = train['date'].dt.dayofweek 
# creating records for all items, in all markets on all dates

# for correct calculation of daily unit sales averages.

u_dates = train.date.unique()

u_stores = train.store_nbr.unique()

u_items = train.item_nbr.unique()

train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)

train = train.reindex(

    pd.MultiIndex.from_product(

        (u_dates, u_stores, u_items),

        names=['date','store_nbr','item_nbr']

    )

)
train.loc[:, 'unit_sales'].fillna(0, inplace=True) # fill NaNs

train.reset_index(inplace=True) # reset index and restoring unique columns  

lastdate = train.iloc[train.shape[0]-1].date
train.head()
tmp = train[['item_nbr','store_nbr','dow','unit_sales']]

ma_dw = tmp.groupby(['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw')

ma_dw.reset_index(inplace=True)

ma_dw.head()
tmp = ma_dw[['item_nbr','store_nbr','madw']]

ma_wk = tmp.groupby(['item_nbr', 'store_nbr'])['madw'].mean().to_frame('mawk')

ma_wk.reset_index(inplace=True)

ma_wk.head()
tmp = train[['item_nbr','store_nbr','unit_sales']]

ma_is = tmp.groupby(['item_nbr', 'store_nbr'])['unit_sales'].mean().to_frame('mais226')
for i in [112,56,28,14,7,3,1]:

    tmp = train[train.date>lastdate-timedelta(int(i))]

    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))

    ma_is = ma_is.join(tmpg, how='left')



del tmp,tmpg
ma_is['mais']=ma_is.median(axis=1)

ma_is.reset_index(inplace=True)
ma_is.head()
def data_to_npreader(store_nbr: int, item_nbr: int) -> NumpyReader:

    unit_sales = train[np.logical_and(train["store_nbr"] == store_nbr,

                                      train['item_nbr'] == item_nbr)].unit_sales



    x = np.asarray(range(len(unit_sales)))

    y = np.asarray(unit_sales)



    dataset = {

        tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,

        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,

    }



    reader = NumpyReader(dataset)

    return x, y, reader
x, y, reader = data_to_npreader(store_nbr=1, item_nbr=105574)



train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(

        reader, batch_size=32, window_size=40)



ar = tf.contrib.timeseries.ARRegressor(

    periodicities=21, input_window_size=30, output_window_size=10,

    num_features=1,

    loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS

)



ar.train(input_fn=train_input_fn, steps=16000)
evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)

# keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']

evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)



(ar_predictions,) = tuple(ar.predict(

    input_fn=tf.contrib.timeseries.predict_continuation_input_fn(

        evaluation, steps=16)))
plt.figure(figsize=(15, 5))

plt.plot(x.reshape(-1), y.reshape(-1), label='origin')

plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')

plt.plot(ar_predictions['times'].reshape(-1), ar_predictions['mean'].reshape(-1), label='prediction')

plt.xlabel('time_step')

plt.ylabel('values')

plt.legend(loc=4)

plt.show()
class _LSTMModel(tfts_model.SequentialTimeSeriesModel):

    """A time series model-building example using an RNNCell."""

    

    def __init__(self, num_units, num_features, dtype=tf.float32):

        """Initialize/configure the model object.

        Note that we do not start graph building here. Rather, this object is a

        configurable factory for TensorFlow graphs which are run by an Estimator.

        Args:

          num_units: The number of units in the model's LSTMCell.

          num_features: The dimensionality of the time series (features per

            timestep).

          dtype: The floating point data type to use.

        """

        

        super(_LSTMModel, self).__init__(

            # Pre-register the metrics we'll be outputting (just a mean here).

            train_output_names=["mean"],

            predict_output_names=["mean"],

            num_features=num_features,

            dtype=dtype)

        self._num_units = num_units

        # Filled in by initialize_graph()

        self._lstm_cell = None

        self._lstm_cell_run = None

        self._predict_from_lstm_output = None



    def initialize_graph(self, input_statistics):

        """Save templates for components, which can then be used repeatedly.

        This method is called every time a new graph is created. It's safe to start

        adding ops to the current default graph here, but the graph should be

        constructed from scratch.

        Args:

          input_statistics: A math_utils.InputStatistics object.

        """

        

        super(_LSTMModel, self).initialize_graph(input_statistics=input_statistics)

        self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)

        # Create templates so we don't have to worry about variable reuse.

        self._lstm_cell_run = tf.make_template(

            name_="lstm_cell",

            func_=self._lstm_cell,

            create_scope_now_=True)

        # Transforms LSTM output into mean predictions.

        self._predict_from_lstm_output = tf.make_template(

            name_="predict_from_lstm_output",

            func_=

            lambda inputs: tf.layers.dense(inputs=inputs, units=self.num_features),

            create_scope_now_=True)



    def get_start_state(self):

        """Return initial state for the time series model."""

        return (

            # Keeps track of the time associated with this state for error checking.

            tf.zeros([], dtype=tf.int64),

            # The previous observation or prediction.

            tf.zeros([self.num_features], dtype=self.dtype),

            # The state of the RNNCell (batch dimension removed since this parent

            # class will broadcast).

            [tf.squeeze(state_element, axis=0)

             for state_element

             in self._lstm_cell.zero_state(batch_size=1, dtype=self.dtype)])



    def _filtering_step(self, current_times, current_values, state, predictions):

        """Update model state based on observations.

        Note that we don't do much here aside from computing a loss. In this case

        it's easier to update the RNN state in _prediction_step, since that covers

        running the RNN both on observations (from this method) and our own

        predictions. This distinction can be important for probabilistic models,

        where repeatedly predicting without filtering should lead to low-confidence

        predictions.

        Args:

          current_times: A [batch size] integer Tensor.

          current_values: A [batch size, self.num_features] floating point Tensor

            with new observations.

          state: The model's state tuple.

          predictions: The output of the previous `_prediction_step`.

        Returns:

          A tuple of new state and a predictions dictionary updated to include a

          loss (note that we could also return other measures of goodness of fit,

          although only "loss" will be optimized).

        """

        state_from_time, prediction, lstm_state = state

        with tf.control_dependencies(

            [tf.assert_equal(current_times, state_from_time)]):

          # Subtract the mean and divide by the variance of the series.  Slightly

          # more efficient if done for a whole window (using the normalize_features

          # argument to SequentialTimeSeriesModel).

          transformed_values = self._scale_data(current_values)

          # Use mean squared error across features for the loss.

          predictions["loss"] = tf.reduce_mean(

              (prediction - transformed_values) ** 2, axis=-1)

          # Keep track of the new observation in model state. It won't be run

          # through the LSTM until the next _imputation_step.

          new_state_tuple = (current_times, transformed_values, lstm_state)

        return (new_state_tuple, predictions)



    def _prediction_step(self, current_times, state):

        """Advance the RNN state using a previous observation or prediction."""

        _, previous_observation_or_prediction, lstm_state = state

        lstm_output, new_lstm_state = self._lstm_cell_run(

            inputs=previous_observation_or_prediction, state=lstm_state)

        next_prediction = self._predict_from_lstm_output(lstm_output)

        new_state_tuple = (current_times, next_prediction, new_lstm_state)

        return new_state_tuple, {"mean": self._scale_back_data(next_prediction)}



    def _imputation_step(self, current_times, state):

        """Advance model state across a gap."""

        # Does not do anything special if we're jumping across a gap. More advanced

        # models, especially probabilistic ones, would want a special case that

        # depends on the gap size.

        return state



    def _exogenous_input_step(

        self, current_times, current_exogenous_regressors, state):

        """Update model state based on exogenous regressors."""

        raise NotImplementedError(

            "Exogenous inputs are not implemented for this example.")
x, y, reader = data_to_npreader(store_nbr=2, item_nbr=105574)



train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(

      reader, batch_size=16, window_size=21)



estimator = tfts_estimators.TimeSeriesRegressor(

      model=_LSTMModel(num_features=1, num_units=32),

      optimizer=tf.train.AdamOptimizer(0.001))



estimator.train(input_fn=train_input_fn, steps=16000)

evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)

evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
(lstm_predictions,) = tuple(estimator.predict(

      input_fn=tf.contrib.timeseries.predict_continuation_input_fn(

          evaluation, steps=16)))
plt.figure(figsize=(15, 5))

plt.plot(x.reshape(-1), y.reshape(-1), label='origin')

plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')

plt.plot(lstm_predictions['times'].reshape(-1), lstm_predictions['mean'].reshape(-1), label='prediction')

plt.xlabel('time_step')

plt.ylabel('values')

plt.legend(loc=4)

plt.show()
# Read test dataset

test = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date'])

test['dow'] = test['date'].dt.dayofweek



# Moving Average

test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])

test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])

test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])

test['unit_sales'] = test.mais



# Autoregressive

ar_predictions['mean'][ar_predictions['mean'] < 0] = 0

test.loc[np.logical_and(test['store_nbr'] == 1, test['item_nbr'] == 105574), 'unit_sales'] = ar_predictions['mean']



# LSTM

lstm_predictions['mean'][lstm_predictions['mean'] < 0] = 0

test.loc[np.logical_and(test['store_nbr'] == 2, test['item_nbr'] == 105574), 'unit_sales'] = lstm_predictions['mean']
pos_idx = test['mawk'] > 0

test_pos = test.loc[pos_idx]

test.loc[pos_idx, 'unit_sales'] = test_pos['unit_sales'] * test_pos['madw'] / test_pos['mawk']
test.loc[:, "unit_sales"].fillna(0, inplace=True)

test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1) # restoring unit values 

test['mais'] = test['mais'].apply(pd.np.expm1) # restoring unit values 
holiday = pd.read_csv('../input/holidays_events.csv', parse_dates=['date'])

holiday = holiday.loc[holiday['transferred'] == False]



test = pd.merge(test, holiday, how = 'left', on =['date'] )

test['transferred'].fillna(True, inplace=True)



test.loc[test['transferred'] == False, 'unit_sales'] *= 1.2

test.loc[test['onpromotion'] == True, 'unit_sales'] *= 1.5
test.loc[np.logical_and(test['store_nbr'] == 1, test['item_nbr'] == 105574)]
test[['id','unit_sales']].to_csv('submission.csv.gz', index=False, compression='gzip')