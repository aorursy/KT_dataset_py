import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/hta-tagging/train.csv')
texts = []
for i in range(len(df['Filename'])):
    fn = df['Filename'][i]
    folder = fn.split("-")[0]
    f = open('/kaggle/input/hta-tagging/train-data/train-data/'+folder+'/'+fn, "r")
    texts.append(f.read())

df['text'] = texts
df
intervention = df.copy().drop(['Blinding of Outcome assessment','Classes'], axis=1)
outcome = df.copy().drop(['Blinding of intervention','Classes'], axis=1)
intervention
outcome
!pip install pycaret
from pycaret.nlp import *
inter_exp = setup (data = intervention, target='text', session_id=10)
# interest to try on other topic modeling algorithms??: https://github.com/pycaret/pycaret/blob/master/nlp.py#L549
inter_lda = create_model('lda')
inter_lda
inter_lda_results = assign_model(inter_lda)
inter_lda_results
plot_model(inter_lda, plot = 'bigram')
plot_model(inter_lda, plot = 'topic_distribution')
plot_model(inter_lda, plot = 'tsne')
evaluate_model(inter_lda)
from pycaret.classification import *
inter_lda_results
inter_lda_data = inter_lda_results.drop(['text','Dominant_Topic','Perc_Dominant_Topic'], axis=1)
inter_lda_data
inter_class = setup(data = inter_lda_data, ignore_features=['Filename'], target='Blinding of intervention', train_size=0.7, session_id = 11)
%time compare_models()
# View the abbrevation of model names here: https://github.com/pycaret/pycaret/blob/master/classification.py#L3022
%time lightgbm = create_model("lightgbm")
%time lightgbm_tune = tune_model("lightgbm")
# ensemble_catboost = ensemble_model(catboost, method="Bagging") #or Boosting
rf_tune = tune_model('rf')
gbc_tune = tune_model('gbc')
blend_models = blend_models(estimator_list = [lightgbm_tune, rf_tune, gbc_tune])
plot_model(blend_models, plot='confusion_matrix')
validation_set = predict_model(blend_models)
validation_set = predict_model(lightgbm_tune)
validation_set = predict_model(rf_tune)
validation_set = predict_model(gbc_tune)
ready = finalize_model(blend_models)
# Load test.csv data
df_test = pd.read_csv('/kaggle/input/hta-tagging/test.csv')
texts = []
for i in range(len(df_test['Id'])):
    fn = df_test['Id'][i]
    folder = fn.split("-")[0]
    f = open('/kaggle/input/hta-tagging/test-data/test-data/'+folder+'/'+fn, "r")
    try:
      texts.append(f.read())
    except:
      print("An exception occurred")
      texts.append("")

df_test['text'] = texts
df_test
from pycaret.nlp import *
test_exp = setup (data = df_test, target='text', session_id=30)
test_lda = create_model('lda')
test_lda
test_lda_results = assign_model(test_lda)
test_lda_results
test_lda_data = test_lda_results.drop(['text','Dominant_Topic','Perc_Dominant_Topic'], axis=1)
test_lda_data
# Don't forget that  N: 0, P: 1, Q: 2
test_result = predict_model(ready, data=test_lda_data)#.drop("Blinding of intervention", axis=1))
test_result
test_result['Label'] = test_result['Label'].replace([0,1,2],['N','P','Q'])
test_result
# from pycaret.nlp import *
# outcm_exp = setup (data = outcome, target='text', session_id=20)
# outcm_lda = create_model('lda')
# outcm_lda_results = assign_model(outcm_lda)
# outcm_lda_data = outcm_lda_results.drop(['text','Dominant_Topic','Perc_Dominant_Topic'], axis=1)
# outcm_lda_data
# outcm_class = setup(data = outcm_lda_data, ignore_features=['Filename'], target='Blinding of Outcome assessment', train_size=0.7, session_id = 21)
# %time compare_models()
# %time lightgbm_tune2 = tune_model("lightgbm")
# %time rf_tune2 = tune_model('rf')
# %time gbc_tune2 = tune_model('gbc')
# blend_models2 = blend_models(estimator_list = [lightgbm_tune2, rf_tune2, gbc_tune2])
# plot_model(blend_models2, plot='confusion_matrix')
# ready2 = finalize_model(blend_models2)
# outcm_result = predict_model(ready2, data=outcm_lda_data)
# outcm_result['Label'] = outcm_result['Label'].replace([0,1,2],['N','P','Q'])
# outcm_result
test_result
test_result['Prediction'] = test_result['Label']+test_result['Label']
test_result
out = test_result[['Id', 'Prediction']]
out
out.to_csv("out.csv", index=False)
