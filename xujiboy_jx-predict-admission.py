# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import fastai

print(fastai.version.__version__)

from fastai.tabular import * 



from pathlib import Path

import seaborn as sns



from IPython.display import display

from IPython.display import HTML

import altair as alt

from altair.vega import v3
##-----------------------------------------------------------

# This whole section 

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



#------------------------------------------------ Defs for future rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(json.dumps(paths)),

    "</script>",

    "This code block sets up embedded rendering in HTML output and<br/>",

    "provides the function `render(chart, id='vega-chart')` for use below."

)))
admission = Path('../input')

admission.ls()
df = pd.read_csv(admission / 'Admission_Predict_Ver1.1.csv')

print('Original columns: ', df.columns)

col_name_map = {}

for col in df.columns:

    col_name_map[col] = col.rstrip()

df.rename(columns=col_name_map, inplace=True)

print('Cleaned columns: ', df.columns)

print('Shape of data: ', df.shape)

display(df.head())

print('Are there any missing data?')

display(df.isna().any())
g = sns.PairGrid(df[df.columns[1:]], diag_sharey=False)

g.map_lower(sns.kdeplot)

g.map_upper(sns.scatterplot)

g.map_diag(sns.kdeplot, lw=3)
dep_var = 'Chance of Admit'

cont_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA']

cat_names = ['Research']

procs = [Categorify, Normalize]
data = (

    TabularList.from_df(df, cat_names=cat_names, cont_names=cont_names, procs=procs)

               .split_by_idx(valid_idx=range(400,500))

               .label_from_df(cols=dep_var)

               .databunch()

)
data.show_batch()
# data.train_ds.to_df() is giving error "AttributeError: 'int' object has no attribute 'relative_to'"
import numpy as np

from typing import List, Tuple
def convert_ddl_to_df(ddl:fastai.basic_data.DeviceDataLoader, 

                      cat_names:list, 

                      cont_names:list)->Tuple[pd.DataFrame, pd.DataFrame]:

    ''' Convert a `fastai.basic_data.DeviceDataLoader` instance into 

        two `pandas.DataFrame`s: the features and the target.

    '''

    list_data_array = list()

    for (x_cat, x_cont),y in ddl:

        tmp_array = np.concatenate((np.array(x_cat),np.array(x_cont),np.array(y).reshape(-1,1)), axis=1)

        list_data_array.append(tmp_array)

    data_array = np.concatenate(list_data_array, axis=0)

    

    columns = []

    for names in (cat_names, cont_names, ['target',]):

        columns.extend(names)

    

    df = pd.DataFrame(data_array, columns=columns)

    return df[columns[:-1]], df[['target']]

dataloader_train = data.dl(DatasetType.Train)

dataloader_valid = data.dl(DatasetType.Valid)

X_train, Y_train = convert_ddl_to_df(dataloader_train, cat_names=cat_names, cont_names=cont_names)

X_valid, Y_valid = convert_ddl_to_df(dataloader_valid, cat_names=cat_names, cont_names=cont_names)
learn = tabular_learner(data, layers=[100,100,], metrics=[root_mean_squared_error, mean_squared_logarithmic_error, r2_score])
learn.fit_one_cycle(30, 1e-2)
learn.save('stage-1')
learn.recorder.plot_metrics()
print(learn.summary())
row = df.iloc[-10]

row['Chance of Admit']
learn.predict(row)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error, r2_score
doe_RF = []

for n in range(10,200,20):

    RFR_model = RandomForestRegressor(n_jobs=-1)

    RFR_model.set_params(n_estimators=n)

    RFR_model.fit(X_train, Y_train)

    doe_RF.append(

        (n, mean_squared_log_error(Y_valid, RFR_model.predict(X_valid)), r2_score(Y_valid, RFR_model.predict(X_valid)))

    )

df_doe_RF = pd.DataFrame(doe_RF, columns=['n_estimator','msle','r2_score'])
base=alt.Chart(df_doe_RF).encode(

    alt.X('n_estimator:Q')

)

msle = base.mark_line(color='red').encode(

    alt.Y('msle:Q'),

)

r2 = base.mark_line(color='green').encode(

    alt.Y('r2_score:Q')

)

render(alt.layer(msle,r2).resolve_scale(y='independent'))

display(df_doe_RF)
df_nn_record = pd.DataFrame(np.array(learn.recorder.metrics), columns=['root_mean_squared_error','mean_squared_logarithmic_error','r2_score'])

print(df_nn_record.max().r2_score, df_nn_record.min().mean_squared_logarithmic_error)
print(df_doe_RF.max().r2_score, df_doe_RF.min().msle)
df = pd.DataFrame({'FN':X_train.columns, 

                   'FI':RFR_model.feature_importances_})
def plot_fi(df, f_name, f_imp):

    chart = alt.Chart(df).mark_bar().encode(

        alt.X(f'{f_imp}:Q'),

        alt.Y(f'{f_name}:N', sort=alt.EncodingSortField(

                field=f"{f_imp}",  # The field to use for the sort

                op="sum",  # The operation to run on the field prior to sorting

                order="descending"  # The order to sort in

            ))

    )

    return chart
render(plot_fi(df, 'FN', 'FI'))
import eli5

from eli5.sklearn import PermutationImportance

from sklearn.metrics.scorer import make_scorer
perm_importance = PermutationImportance(RFR_model, scoring=make_scorer(r2_score),

                                   n_iter=50, random_state=42, cv="prefit")

perm_importance.fit(X_valid, Y_valid)

df_imp = eli5.explain_weights_df(perm_importance)

df_label = pd.DataFrame({'feature': [ "x" + str(i) for i in range(len(X_valid.columns))], 'feature_name': X_valid.columns.values})

df_imp = pd.merge(df_label, df_imp, on='feature', how='inner', validate="one_to_one")
render(plot_fi(df_imp, 'feature_name', 'weight'))