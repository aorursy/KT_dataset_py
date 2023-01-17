from google.cloud import bigquery

client = bigquery.Client()

import matplotlib.pyplot as plt

import matplotlib.ticker as mtick

import numpy as np

import pandas as pd

import itertools



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import  confusion_matrix, recall_score,precision_score, precision_recall_curve, f1_score, fbeta_score

from sklearn.utils.fixes import signature

miner_vectors_limit = 2000

non_miner_vectors_limit = 20000
sql='''

WITH 

output_ages AS (

  SELECT

    ARRAY_TO_STRING(outputs.addresses,',') AS output_ages_address,

    MIN(block_timestamp_month) AS output_month_min,

    MAX(block_timestamp_month) AS output_month_max

  FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

  GROUP BY output_ages_address

)

,input_ages AS (

  SELECT

    ARRAY_TO_STRING(inputs.addresses,',') AS input_ages_address,

    MIN(block_timestamp_month) AS input_month_min,

    MAX(block_timestamp_month) AS input_month_max

  FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(inputs) AS inputs

  GROUP BY input_ages_address

)

,output_monthly_stats AS (

  SELECT

    ARRAY_TO_STRING(outputs.addresses,',') AS output_monthly_stats_address, 

    COUNT(DISTINCT block_timestamp_month) AS output_active_months,

    COUNT(outputs) AS total_tx_output_count,

    SUM(value) AS total_tx_output_value,

    AVG(value) AS mean_tx_output_value,

    STDDEV(value) AS stddev_tx_output_value,

    COUNT(DISTINCT(`hash`)) AS total_output_tx,

    SUM(value)/COUNT(block_timestamp_month) AS mean_monthly_output_value,

    COUNT(outputs.addresses)/COUNT(block_timestamp_month) AS mean_monthly_output_count

  FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

  GROUP BY output_monthly_stats_address

)

,input_monthly_stats AS (

  SELECT

    ARRAY_TO_STRING(inputs.addresses,',') AS input_monthly_stats_address, 

    COUNT(DISTINCT block_timestamp_month) AS input_active_months,

    COUNT(inputs) AS total_tx_input_count,

    SUM(value) AS total_tx_input_value,

    AVG(value) AS mean_tx_input_value,

    STDDEV(value) AS stddev_tx_input_value,

    COUNT(DISTINCT(`hash`)) AS total_input_tx,

    SUM(value)/COUNT(block_timestamp_month) AS mean_monthly_input_value,

    COUNT(inputs.addresses)/COUNT(block_timestamp_month) AS mean_monthly_input_count

  FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(inputs) AS inputs

  GROUP BY input_monthly_stats_address

)

,output_idle_times AS (

  SELECT

    address AS idle_time_address,

    AVG(idle_time) AS mean_output_idle_time,

    STDDEV(idle_time) AS stddev_output_idle_time

  FROM

  (

    SELECT 

      event.address,

      IF(prev_block_time IS NULL, NULL, UNIX_SECONDS(block_time) - UNIX_SECONDS(prev_block_time)) AS idle_time

    FROM (

      SELECT

        ARRAY_TO_STRING(outputs.addresses,',') AS address, 

        block_timestamp AS block_time,

        LAG(block_timestamp) OVER (PARTITION BY ARRAY_TO_STRING(outputs.addresses,',') ORDER BY block_timestamp) AS prev_block_time

      FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

    ) AS event

    WHERE block_time != prev_block_time

  )

  GROUP BY address

)

,input_idle_times AS (

  SELECT

    address AS idle_time_address,

    AVG(idle_time) AS mean_input_idle_time,

    STDDEV(idle_time) AS stddev_input_idle_time

  FROM

  (

    SELECT 

      event.address,

      IF(prev_block_time IS NULL, NULL, UNIX_SECONDS(block_time) - UNIX_SECONDS(prev_block_time)) AS idle_time

    FROM (

      SELECT

        ARRAY_TO_STRING(inputs.addresses,',') AS address, 

        block_timestamp AS block_time,

        LAG(block_timestamp) OVER (PARTITION BY ARRAY_TO_STRING(inputs.addresses,',') ORDER BY block_timestamp) AS prev_block_time

      FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(inputs) AS inputs

    ) AS event

    WHERE block_time != prev_block_time

  )

  GROUP BY address

)

--,miners AS (

--)



(SELECT

  TRUE AS is_miner,

  output_ages_address AS address,

  UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) AS output_month_min,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) AS output_month_max,

  UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS input_month_min,

  UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) AS input_month_max,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) AS output_active_time,

  UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS input_active_time,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) AS io_max_lag,

  UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS io_min_lag,

  output_monthly_stats.output_active_months,

  output_monthly_stats.total_tx_output_count,

  output_monthly_stats.total_tx_output_value,

  output_monthly_stats.mean_tx_output_value,

  output_monthly_stats.stddev_tx_output_value,

  output_monthly_stats.total_output_tx,

  output_monthly_stats.mean_monthly_output_value,

  output_monthly_stats.mean_monthly_output_count,

  input_monthly_stats.input_active_months,

  input_monthly_stats.total_tx_input_count,

  input_monthly_stats.total_tx_input_value,

  input_monthly_stats.mean_tx_input_value,

  input_monthly_stats.stddev_tx_input_value,

  input_monthly_stats.total_input_tx,

  input_monthly_stats.mean_monthly_input_value,

  input_monthly_stats.mean_monthly_input_count,

  output_idle_times.mean_output_idle_time,

  output_idle_times.stddev_output_idle_time,

  input_idle_times.mean_input_idle_time,

  input_idle_times.stddev_input_idle_time

FROM

  output_ages, output_monthly_stats, output_idle_times,

  input_ages,  input_monthly_stats, input_idle_times

WHERE TRUE

  AND output_ages.output_ages_address = output_monthly_stats.output_monthly_stats_address

  AND output_ages.output_ages_address = output_idle_times.idle_time_address

  AND output_ages.output_ages_address = input_monthly_stats.input_monthly_stats_address

  AND output_ages.output_ages_address = input_ages.input_ages_address

  AND output_ages.output_ages_address = input_idle_times.idle_time_address

  AND output_ages.output_ages_address IN

(

  SELECT 

    ARRAY_TO_STRING(outputs.addresses,',') AS miner

  FROM 

  `bigquery-public-data.crypto_bitcoin.blocks` AS blocks,

  `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

  WHERE blocks.hash = transactions.block_hash 

    AND is_coinbase IS TRUE

    AND ( FALSE

      --

      -- miner signatures from https://en.bitcoin.it/wiki/Comparison_of_mining_pools

      --

      OR coinbase_param LIKE '%4d696e656420627920416e74506f6f6c%' --AntPool

      OR coinbase_param LIKE '%2f42434d6f6e737465722f%' --BCMonster

      --BitcoinAffiliateNetwork

      OR coinbase_param LIKE '%4269744d696e746572%' --BitMinter

      --BTC.com

      --BTCC Pool

      --BTCDig

      OR coinbase_param LIKE '%2f7374726174756d2f%' --Btcmp

      --btcZPool.com

      --BW Mining

      OR coinbase_param LIKE '%456c6967697573%' --Eligius

      --F2Pool

      --GHash.IO

      --Give Me COINS

      --Golden Nonce Pool

      OR coinbase_param LIKE '%2f627261766f2d6d696e696e672f%' --Bravo Mining

      OR coinbase_param LIKE '%4b616e6f%' --KanoPool

      --kmdPool.org

      OR coinbase_param LIKE '%2f6d6d706f6f6c%' --Merge Mining Pool

      --MergeMining

      --Multipool

      --P2Pool

      OR coinbase_param LIKE '%2f736c7573682f%' --Slush Pool

      --ZenPool.org

    )

  GROUP BY miner

  HAVING COUNT(1) >= 20 

)

LIMIT {})

UNION ALL

(SELECT

  FALSE AS is_miner,

  output_ages_address AS address,

  UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) AS output_month_min,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) AS output_month_max,

  UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS input_month_min,

  UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) AS input_month_max,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) AS output_active_time,

  UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS input_active_time,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) AS io_max_lag,

  UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS io_min_lag,

  output_monthly_stats.output_active_months,

  output_monthly_stats.total_tx_output_count,

  output_monthly_stats.total_tx_output_value,

  output_monthly_stats.mean_tx_output_value,

  output_monthly_stats.stddev_tx_output_value,

  output_monthly_stats.total_output_tx,

  output_monthly_stats.mean_monthly_output_value,

  output_monthly_stats.mean_monthly_output_count,

  input_monthly_stats.input_active_months,

  input_monthly_stats.total_tx_input_count,

  input_monthly_stats.total_tx_input_value,

  input_monthly_stats.mean_tx_input_value,

  input_monthly_stats.stddev_tx_input_value,

  input_monthly_stats.total_input_tx,

  input_monthly_stats.mean_monthly_input_value,

  input_monthly_stats.mean_monthly_input_count,

  output_idle_times.mean_output_idle_time,

  output_idle_times.stddev_output_idle_time,

  input_idle_times.mean_input_idle_time,

  input_idle_times.stddev_input_idle_time

FROM

  output_ages, output_monthly_stats, output_idle_times,

  input_ages,  input_monthly_stats, input_idle_times

WHERE TRUE

  AND output_ages.output_ages_address = output_monthly_stats.output_monthly_stats_address

  AND output_ages.output_ages_address = output_idle_times.idle_time_address

  AND output_ages.output_ages_address = input_monthly_stats.input_monthly_stats_address

  AND output_ages.output_ages_address = input_ages.input_ages_address

  AND output_ages.output_ages_address = input_idle_times.idle_time_address

  AND output_ages.output_ages_address NOT IN

(

  SELECT 

    ARRAY_TO_STRING(outputs.addresses,',') AS miner

  FROM 

  `bigquery-public-data.crypto_bitcoin.blocks` AS blocks,

  `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

  WHERE blocks.hash = transactions.block_hash 

    AND is_coinbase IS TRUE

    AND ( FALSE

      --

      -- miner signatures from https://en.bitcoin.it/wiki/Comparison_of_mining_pools

      --

      OR coinbase_param LIKE '%4d696e656420627920416e74506f6f6c%' --AntPool

      OR coinbase_param LIKE '%2f42434d6f6e737465722f%' --BCMonster

      --BitcoinAffiliateNetwork

      OR coinbase_param LIKE '%4269744d696e746572%' --BitMinter

      --BTC.com

      --BTCC Pool

      --BTCDig

      OR coinbase_param LIKE '%2f7374726174756d2f%' --Btcmp

      --btcZPool.com

      --BW Mining

      OR coinbase_param LIKE '%456c6967697573%' --Eligius

      --F2Pool

      --GHash.IO

      --Give Me COINS

      --Golden Nonce Pool

      OR coinbase_param LIKE '%2f627261766f2d6d696e696e672f%' --Bravo Mining

      OR coinbase_param LIKE '%4b616e6f%' --KanoPool

      --kmdPool.org

      OR coinbase_param LIKE '%2f6d6d706f6f6c%' --Merge Mining Pool

      --MergeMining

      --Multipool

      --P2Pool

      OR coinbase_param LIKE '%2f736c7573682f%' --Slush Pool

      --ZenPool.org

    )

  GROUP BY miner

  HAVING COUNT(1) >= 20 

)

LIMIT {})

'''.format(miner_vectors_limit, non_miner_vectors_limit)
df = client.query(sql).to_dataframe()
df.info()
#drop columns with null values

df.drop(labels=['stddev_output_idle_time','stddev_input_idle_time'], axis=1, inplace=True)
#get rid of non-numeric features

features = df.drop(labels=['is_miner','address'], axis=1)

target = df['is_miner'].values

indices = range(len(features))



#Train test split

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, target, indices,  test_size=0.2)
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced')

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test) #

probs = rf.predict_proba(X_test)[:,1] #positive class probabilities
precision, recall, thresholds = precision_recall_curve(y_test, probs)
# Precision / recall curve code adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html



fig, ax = plt.subplots(figsize=(8,6))

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.0])

plt.xlim([0.0, 1.0])

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.title('Mining Pool Detector - Precision/Recall Curve', fontsize=14)
#confusion matrix code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)

    dummy=np.array([[0,0],[0,0]])

    plt.figure(figsize=(8,6))

    plt.imshow(dummy, interpolation='nearest', cmap=cmap)

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()





# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred)

class_names = ['not mining pool', 'mining pool']

np.set_printoptions(precision=2)



# Plot confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,

                      title='Mining Pool Detector - Confusion Matrix')



plt.show()
x_pos = np.arange(len(features.columns))

btc_importances = rf.feature_importances_



inds = np.argsort(btc_importances)[::-1]

btc_importances = btc_importances[inds]

cols = features.columns[inds]

bar_width = .8



#how many features to plot?

n_features=12

x_pos = x_pos[:n_features][::-1]

btc_importances = btc_importances[:n_features]



#plot

plt.figure(figsize=(12,6))

plt.barh(x_pos, btc_importances, bar_width, label='BTC model')

plt.yticks(x_pos, cols, rotation=0, fontsize=14)

plt.xlabel('feature importance', fontsize=14)

plt.title('Mining Pool Detector', fontsize=20)

plt.tight_layout()
#data points where model predicts true, but are labelled as false

false_positives = (y_test==False) & (y_pred==True)
#subset to test set data only

df_test = df.iloc[indices_test, :]



print('False Positive addresses')



#subset test set to false positives only

df_test.iloc[false_positives].head(15)