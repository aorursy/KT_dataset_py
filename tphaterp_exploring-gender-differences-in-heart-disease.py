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
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
# Read in the kaggle dataset

heart_df = pd.read_csv("/kaggle/input/heart-disease-dataset/heart.csv")

heart_df.head()
heart_df.hist(figsize = (15,15))
# Check for null values

heart_df.isnull().sum()
X_training, X_vault, y_training, y_vault = train_test_split(heart_df.drop(columns = 'target'),

                                                            heart_df[['target']],

                                                            test_size = 0.2, 

                                                            random_state = 1)
# The following code was taken from (https://www.kaggle.com/jakevdp/altair-kaggle-renderer, 

# attributed to the user"no data sources"). This code allows altair charts to render properly

# within the kaggle kernel.



import altair as alt

import json

from IPython.display import HTML



KAGGLE_HTML_TEMPLATE = """

<style>

.vega-actions a {{

    margin-right: 12px;

    color: #757575;

    font-weight: normal;

    font-size: 13px;

}}

.error {{

    color: red;

}}

</style>

<div id="{output_div}"></div>

<script>

requirejs.config({{

    "paths": {{

        "vega": "{base_url}/vega@{vega_version}?noext",

        "vega-lib": "{base_url}/vega-lib?noext",

        "vega-lite": "{base_url}/vega-lite@{vegalite_version}?noext",

        "vega-embed": "{base_url}/vega-embed@{vegaembed_version}?noext",

    }}

}});

function showError(el, error){{

    el.innerHTML = ('<div class="error">'

                    + '<p>JavaScript Error: ' + error.message + '</p>'

                    + "<p>This usually means there's a typo in your chart specification. "

                    + "See the javascript console for the full traceback.</p>"

                    + '</div>');

    throw error;

}}

require(["vega-embed"], function(vegaEmbed) {{

    const spec = {spec};

    const embed_opt = {embed_opt};

    const el = document.getElementById('{output_div}');

    vegaEmbed("#{output_div}", spec, embed_opt)

      .catch(error => showError(el, error));

}});

</script>

"""



class KaggleHtml(object):

    def __init__(self, base_url='https://cdn.jsdelivr.net/npm'):

        self.chart_count = 0

        self.base_url = base_url

        

    @property

    def output_div(self):

        return "vega-chart-{}".format(self.chart_count)

        

    def __call__(self, spec, embed_options=None, json_kwds=None):

        # we need to increment the div, because all charts live in the same document

        self.chart_count += 1

        embed_options = embed_options or {}

        json_kwds = json_kwds or {}

        html = KAGGLE_HTML_TEMPLATE.format(

            spec=json.dumps(spec, **json_kwds),

            embed_opt=json.dumps(embed_options),

            output_div=self.output_div,

            base_url=self.base_url,

            vega_version=alt.VEGA_VERSION,

            vegalite_version=alt.VEGALITE_VERSION,

            vegaembed_version=alt.VEGAEMBED_VERSION

        )

        return {"text/html": html}

    

alt.renderers.register('kaggle', KaggleHtml())

print("Define and register the kaggle renderer. Enable with\n\n"

      "    alt.renderers.enable('kaggle')")
alt.renderers.enable('kaggle')

# https://altair-viz.github.io/gallery/scatter_matrix.html

alt.Chart(X_training).mark_circle().encode(

    alt.X(alt.repeat("column"), type = "quantitative"),

    alt.Y(alt.repeat("row"), type = "quantitative"),

    alt.Color("sex:N")

).properties(

    width = 120,

    height = 120

).repeat(

    row = ["age", "chol", "oldpeak","thalach", "trestbps"],

    column = ["trestbps", "thalach", "oldpeak", "chol", "age"]

)
# Combine the features and target of the training dataset 

# (Note: I didn't do this on the original dataframe to avoid bias entering our analysis)

pd.concat([X_training, y_training], sort = False, axis = 1)
# .corr() finds correlational relationships for each column

# I use reset_index() for the melting step to treat column names as a variable instead of an index (see below)

pd.concat([X_training, y_training], sort = False, axis = 1).corr().reset_index()
# added rounding and melt data frame columns into "var2"

pd.concat([X_training, y_training], sort = False, axis = 1).corr().round(2).reset_index().melt(id_vars = "index",var_name = "var2", value_name = "corr_val")
# Inspired by https://altair-viz.github.io/gallery/layered_heatmap_text.html

def make_heatmap(corr_df):

    '''

    Take in a correlational dataframe and create a heatmap

    

    Arguments: 

    corr_df (DataFrame) - Dataframe of correlational values

    '''

     

    base = alt.Chart(corr_df).encode(

        alt.X("index"),

        alt.Y("var2")

    )

    heatmap = base.mark_rect().encode(

        alt.Color("corr_val", scale = alt.Scale(scheme = "viridis"))

    )

    

    text = base.mark_text().encode(

    text = "corr_val"

    )

    return (heatmap + text).properties(height = 500, width = 500)


corr_df = pd.concat([X_training, y_training], sort = False, axis = 1).corr().round(2).reset_index().melt(id_vars = "index",var_name = "var2", value_name = "corr_val")

make_heatmap(corr_df)
male_corr = pd.concat([X_training, y_training],

                      sort = False, 

                      axis = 1).query("sex == 1").drop(["sex"], 

                                                       axis = 1).corr().round(2).reset_index().melt(id_vars = "index",var_name = "var2", value_name = "corr_val")

make_heatmap(male_corr)
female_corr = pd.concat([X_training, y_training],

                      sort = False, 

                      axis = 1).query("sex == 0").drop(["sex"], 

                                                       axis = 1).corr().round(2).reset_index().melt(id_vars = "index",var_name = "var2", value_name = "corr_val")

make_heatmap(female_corr)
heart_df.query("sex == 1").shape
heart_df.query("sex == 0").shape
X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size = 0.2, random_state = 2)



numerical_categories = ["age", "chol", "oldpeak", "thalach", "trestbps"]



preprocessor = ColumnTransformer(transformers=[

        ('scale', StandardScaler(), numerical_categories)], remainder = "passthrough")



new_columns = ["age", "chol", "oldpeak", "thalach", "trestbps", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]



X_train_scaled = pd.DataFrame(preprocessor.fit_transform(X_train), columns = new_columns)

X_test_scaled = pd.DataFrame(preprocessor.transform(X_test), columns = new_columns)
X_train_comp = X_train_scaled[["thalach", "oldpeak","exang","cp","ca","thal","slope"]]

X_test_comp = X_test_scaled[["thalach", "oldpeak","exang","cp","ca","thal","slope"]]
comp_RF_model = RandomForestClassifier(n_estimators = 150)

comp_RF_model.fit(X_train_comp, y_train.to_numpy().ravel())

print(f"Training Accuracy: {comp_RF_model.score(X_train_comp, y_train)}")

print(f"Test Accuracy: {comp_RF_model.score(X_test_comp, y_test)}")
comp_XGBoost = XGBClassifier()

comp_XGBoost.fit(X_train_comp, y_train.to_numpy().ravel())

print(f"Training Accuracy: {comp_XGBoost.score(X_train_comp, y_train)}")

print(f"Test Accuracy: {comp_XGBoost.score(X_test_comp, y_test)}")
X_train_male = X_train_scaled[["thalach", "oldpeak","exang","cp","slope","ca","age"]]

X_test_male = X_test_scaled[["thalach", "oldpeak","exang","cp","slope","ca","age"]]
male_RF_model = RandomForestClassifier(n_estimators = 150)

male_RF_model.fit(X_train_male, y_train.to_numpy().ravel())

print(f"Training Accuracy: {male_RF_model.score(X_train_male, y_train)}")

print(f"Test Accuracy: {male_RF_model.score(X_test_male, y_test)}")
male_XGBoost = XGBClassifier()

male_XGBoost.fit(X_train_male, y_train.to_numpy().ravel())

print(f"Training Accuracy: {male_XGBoost.score(X_train_male, y_train)}")

print(f"Test Accuracy: {male_XGBoost.score(X_test_male, y_test)}")
X_train_female = X_train_scaled[["cp", "exang","thal","oldpeak","ca","slope","trestbps"]]

X_test_female = X_test_scaled[["cp", "exang","thal","oldpeak","ca","slope","trestbps"]]
female_RF_model = RandomForestClassifier(n_estimators = 150)

female_RF_model.fit(X_train_female, y_train.to_numpy().ravel())

print(f"Training Accuracy: {female_RF_model.score(X_train_female, y_train)}")

print(f"Test Accuracy: {female_RF_model.score(X_test_female, y_test)}")
female_XGBoost = XGBClassifier()

female_XGBoost.fit(X_train_female, y_train.to_numpy().ravel())

print(f"Training Accuracy: {female_XGBoost.score(X_train_female, y_train)}")

print(f"Test Accuracy: {female_XGBoost.score(X_test_female, y_test)}")
vault_male = pd.concat([X_vault, y_vault], axis = 1, sort = False).query("sex == 1")

vault_female = pd.concat([X_vault, y_vault], axis = 1, sort = False).query("sex == 0")

print(vault_male.shape)

print(vault_female.shape)
# Apply the scaling

X_vault_scaled = pd.DataFrame(preprocessor.transform(X_vault), columns = new_columns)

X_vault_male_scaled = pd.DataFrame(preprocessor.transform(vault_male.drop(columns = ['target'])), columns = new_columns)

X_vault_female_scaled = pd.DataFrame(preprocessor.transform(vault_female.drop(columns = ['target'])), columns = new_columns)
comp_features = ["thalach", "oldpeak","exang","cp","ca","thal","slope"]

male_features = ["thalach", "oldpeak","exang","cp","slope","ca","age"]

female_features = ["cp", "exang","thal","oldpeak","ca","slope","trestbps"]



XG_score_dict = {"XG Boost Scores":["Both Genders", "Male Data","Female Data"]}



XG_score_dict["Comprehensive Model Acc"] = [comp_XGBoost.score(X_vault_scaled[comp_features], y_vault),

                                        comp_XGBoost.score(X_vault_male_scaled[comp_features], vault_male[['target']]),

                                       comp_XGBoost.score(X_vault_female_scaled[comp_features], vault_female[['target']])]



XG_score_dict["Male-focused Model Acc"] = [male_XGBoost.score(X_vault_scaled[male_features], y_vault),

                                        male_XGBoost.score(X_vault_male_scaled[male_features], vault_male[['target']]),

                                       male_XGBoost.score(X_vault_female_scaled[male_features], vault_female[['target']])]



XG_score_dict["Female-focused Model Acc"] =[female_XGBoost.score(X_vault_scaled[female_features], y_vault),

                                        female_XGBoost.score(X_vault_male_scaled[female_features], vault_male[['target']]),

                                       female_XGBoost.score(X_vault_female_scaled[female_features], vault_female[['target']])]

pd.DataFrame(XG_score_dict).round(2)
RF_score_dict = {"RF Scores":["Both Genders", "Male Data","Female Data"]}



RF_score_dict["Comprehensive Model Acc"] = [comp_RF_model.score(X_vault_scaled[comp_features], y_vault),

                                        comp_RF_model.score(X_vault_male_scaled[comp_features], vault_male[['target']]),

                                       comp_RF_model.score(X_vault_female_scaled[comp_features], vault_female[['target']])]



RF_score_dict["Male-focused Model Acc"] = [male_RF_model.score(X_vault_scaled[male_features], y_vault),

                                        male_RF_model.score(X_vault_male_scaled[male_features], vault_male[['target']]),

                                       male_RF_model.score(X_vault_female_scaled[male_features], vault_female[['target']])]



RF_score_dict["Female-focused Model Acc"] =[female_RF_model.score(X_vault_scaled[female_features], y_vault),

                                        female_RF_model.score(X_vault_male_scaled[female_features], vault_male[['target']]),

                                       female_RF_model.score(X_vault_female_scaled[female_features], vault_female[['target']])]



pd.DataFrame(RF_score_dict).round(2)