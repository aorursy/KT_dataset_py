# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

covid_19_clean_complete = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
covid_19_clean_complete['Date'] = pd.to_datetime(covid_19_clean_complete['Date'], format="%m/%d/%y")
countries = covid_19_clean_complete['Country/Region'].unique()
countries
country_series = {c:covid_19_clean_complete[covid_19_clean_complete['Country/Region'] == c].groupby('Date').agg('sum').drop(['Lat', 'Long'], axis=1) for c in countries}
#np.log10(country_series['Mainland China']['Confirmed'].values)
c_confirmed = {c:(country_series[c]['Confirmed'] + 1).pct_change(fill_method='bfill').fillna(0) for c in countries}

c_deaths = {c:(country_series[c]['Deaths'] + 1).pct_change(fill_method='bfill').fillna(0) for c in countries}
import tensorflow as tf

import tensorflow_probability as tfp

import tensorflow.keras as tfk

import tensorflow.keras.layers as tfkl

tfpl = tfp.layers
class SpreadModel(tfk.Model):

    

    def __init__(self,

                 n_outputs,

                 **kwargs):

        

        super(SpreadModel, self).__init__(**kwargs)

        self.input_layer = tfkl.Dense(n_outputs)

        self.lstms = [tfkl.LSTM(256, return_sequences=True, return_state=True) for i in range(4)]

        event_shape = [n_outputs]

        num_components = 20

        params_size = tfpl.MixtureSameFamily.params_size(

            num_components,

            component_params_size=tfpl.IndependentNormal.params_size(event_shape))

        self.output_dense = tfkl.Dense(params_size, activation=None)

        self.output_layer = tfpl.MixtureSameFamily(num_components, tfpl.IndependentNormal(event_shape))

        

        

    def call(self, inputs, hidden_states=None, training=None):

        

        x = self.input_layer(inputs, training=training)

        

        next_hidden_states = []

        if hidden_states is not None:

            

            for rnn, hs in zip(self.lstms, hidden_states):

                x, fms, fcs = rnn(x, training=training, initial_state=hs)

                next_hidden_states.append((fms, fcs))

        else:

            for rnn in self.lstms:

                x, fms, fcs = rnn(x, training=training)

                next_hidden_states.append((fms, fcs))

        dist_params = self.output_dense(x)

        return self.output_layer(dist_params, training=training), next_hidden_states

        
spreadModel = SpreadModel(1)
import numpy as np





#Remove countries that aren't testing



filter_countries = set([

    'China',

    'Japan',

    'Hong Kong',

    'Singapore',

    'Italy',

    'France',

    'UK',

    'Iceland',

    'Japan',

    'Switzerland',

    'South Korea',

    'Taiwan',

    'Spain',

    'Australia',

    'Finland',

    'Sweden',

    'Norway',

    'Germany',

    'Canada',

])
data_filtered = {k:v for k,v in c_confirmed.items() if k in filter_countries}
c_confirmed.keys()
data = np.clip(np.expand_dims(np.asarray([vs.values for vs in data_filtered.values()]), axis=-1).astype(np.float32), 0., 1.)
data_x = data[:,:-1,:]

data_y = data[:,1:,:]
ds = tf.data.Dataset.from_tensor_slices((data_x, data_y))

ds = ds.batch(64)
opt = tfk.optimizers.Adam(1e-6)
@tf.function

def train_step(data):

    

    with tf.GradientTape() as g:

        

        inputs, targets = data

        

        pred_dist, _ = spreadModel(inputs, training=True)

        

        loss = tf.reduce_mean(-pred_dist.log_prob(targets))

        

    grads = g.gradient(loss, spreadModel.trainable_variables)

    grads, _ = tf.clip_by_global_norm(grads, 1.0)

    opt.apply_gradients(zip(grads, spreadModel.trainable_variables))

    

    return loss
for epoch in range(5000):

    it = iter(ds)

    for i, data in enumerate(it):

        #print(data)

        loss = train_step(data).numpy()

        if loss < 0.0001:

            break

    if epoch % 100 == 0:

        print(loss)

    if loss < 0.0001:

        break
def predict_next_month(country):

    dist, states = spreadModel(np.expand_dims(np.expand_dims(c_confirmed[country].values, axis=0), axis=-1), training=False)

    smpls = dist.sample()

    #print(smpls)

    #raise Exception()

    #am = tf.argmax(dist.log_prob(smpls)[...,-1])[0].numpy()

    #good_sample = smpls[am, 0, -1]

    nxt_sample = tf.expand_dims(tf.expand_dims(tf.expand_dims(smpls[0,-1,0], axis=-1),axis=0),axis=0)

    nxt_sample

    steps = [nxt_sample]

    

    for i in range(29):

        dist, states = spreadModel(nxt_sample, training=False, hidden_states=states)

        smpls = dist.sample()

        #raise Exception()

        #good_sample = smpls[am, 0, -1]

        #nxt_sample = tf.expand_dims(tf.expand_dims(good_sample, axis=0),axis=-1)

        steps.append(smpls)

        

    return country_series[country]['Confirmed'][-1] * np.cumprod(1 + np.clip(tf.squeeze(tf.concat(steps, axis=1)).numpy().astype(np.float128), 0., 10.))
def sample_multi_next_month(country, samples=20):

    smpls = []

    for i in range(samples):

        r = predict_next_month(country)

        smpls.append(r)

    return smpls
futures = sample_multi_next_month('US')

future_means = np.stack(futures, axis=0).mean(axis=0).astype(np.int32).tolist()
print('USA confirmed coming month:\n', pd.Series(data=future_means, index=pd.date_range(c_confirmed['Sweden'].reset_index()['Date'].tolist()[-1] + pd.Timedelta('1 day'), periods=30, freq='D')))
futures = sample_multi_next_month('Sweden')

future_means = np.stack(futures, axis=0).mean(axis=0).astype(np.int32).tolist()
print('Sweden confirmed coming month:\n', pd.Series(data=future_means, index=pd.date_range(c_confirmed['Sweden'].reset_index()['Date'].tolist()[-1] + pd.Timedelta('1 day'), periods=30, freq='D')))