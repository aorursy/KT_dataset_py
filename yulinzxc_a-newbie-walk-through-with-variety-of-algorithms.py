prefix =  "../input/costa-rican-household-poverty-prediction/"
from __future__ import division
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
# link pandas and plotly
import cufflinks as cf

init_notebook_mode(connected=True)
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
%matplotlib inline

from matplotlib import rc
def plot_hist(x, col_name, title, x_label=None, histnorm="percent"):
    if x_label == None:
        x_label = col_name
    data = []
    for i in range(4):
        trace = go.Histogram(
            x=x[x.Target == i + 1][col_name],
            opacity=0.75,
            histnorm=histnorm,
            name=poverty_map[i+1]
        )
        
        data.append(trace)
        
    layout = go.Layout(
        barmode='stack', 
        title=title,
        xaxis=dict(
            title=x_label
        ),
        yaxis=dict(
            title="Percentage(%)"
        )
    )
    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename='overlaid histogram')
def pca_3d_plot(pca_train_df):
    data = []
    colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)', 'rgb(230, 230, 20)']

    for i in range(4):
        color = colors[i]
        x = pca_train_df[pca_train_df.Target == i + 1][0]
        y = pca_train_df[pca_train_df.Target == i + 1][1]
        z = pca_train_df[pca_train_df.Target == i + 1][2]

        trace = dict(
            name = poverty_map[i + 1],
            x = x, y = y, z = z,
            type = "scatter3d",    
            mode = 'markers',
            marker = dict( size=3, color=color, line=dict(width=0) ) )
        data.append( trace )

    layout = dict(
        width=800,
        height=550,
        autosize=False,
        title='Costa Rican Poverty Prediction PCA',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio = dict( x=1, y=1, z=0.7 ),
            aspectmode = 'manual'        
        ),
    )

    fig = dict(data=data, layout=layout)

    # IPython notebook
    iplot(fig, filename='costa_rican', validate=False)
from sklearn.metrics import silhouette_score, make_scorer
def search_optimised_k(n_clusters_list, random_state, X):
    silhouette_score_list = []
    result_dict  = {}
    for n_clusters in n_clusters_list:
        km_predict = KMeans(random_state=random_state, n_jobs=-1, n_clusters=n_clusters).fit_predict(X)
        silhouette_score_list.append(silhouette_score(X=X, labels=km_predict, random_state=16446054))
    result_dict["silhouette_score"] = silhouette_score_list
    result_dict["n_clusters"] = n_clusters_list
    
    return result_dict
        
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


class knn_agent:
    def __init__(self):
        self.uniform_neighbours_num = None
        self.distance_neighbours_num = None
        # === models
        self.uniform_knn = None
        self.distance_knn = None
        
        self.f1_df = None
        

    def optimise_params(self, X_train, y_train, X_dev, y_dev, max_k_num):
        uniform_f1_test_list = []
        distance_f1_test_list = []
        
        for i in range(1, max_k_num):
            uniform_knn = KNeighborsClassifier(n_neighbors=i, weights="uniform")
            distance_knn = KNeighborsClassifier(n_neighbors=i, weights="distance")

            uniform_prediction = uniform_knn.fit(X_train, y_train).predict(X_dev)
            distance_prediction = distance_knn.fit(X_train, y_train).predict(X_dev)

            uniform_f1_test_list.append(f1_score(y_true=y_dev, y_pred=uniform_prediction, average='macro'))
            distance_f1_test_list.append(f1_score(y_true=y_dev, y_pred=distance_prediction, average='macro'))

        self.f1_df = pd.DataFrame(data={"F1(uniform)": uniform_f1_test_list,
                                       "F1(distance)": distance_f1_test_list},
                                 index=range(1, max_k_num))

        # Observe from the plot we can tell that the global minimum is the uniform neighbours with total market values
        self.uniform_neighbours_num = self.f1_df.sort_values("F1(uniform)", ascending=False).head(1).index.values.item(0)

        self.distance_neighbours_num = self.f1_df.sort_values("F1(distance)", ascending=False).head(1).index.values.item(0)

        # Initialise the model with optimised k number
        self.uniform_knn = KNeighborsClassifier(n_neighbors=self.uniform_neighbours_num, weights="uniform")
        self.distance_knn = KNeighborsClassifier(n_neighbors=self.distance_neighbours_num, weights="distance")
        
        self.plot_validation_result()

    def plot_validation_result(self):
        
        self.f1_df.iplot(kind='scatter', yTitle="F1 Score", xTitle="K number", title="Optimised K Number Searching")
        
        
    # Noted that predict data need to clean before feeding
    def predict(self, X_train, y_train, X_predict):
        return (self.uniform_knn.fit(X_train, y_train).predict(X_predict), self.distance_knn.fit(X_train, y_train).predict(X_predict))

    def evaluate(self, X_train, y_train, X_test, y_test):
        uniform_prediction = self.uniform_knn.fit(X_train, y_train).predict(X_test)

        distance_prediction = self.distance_knn.fit(X_train, y_train).predict(X_test)

        return (f1_score(y_true=y_test, y_pred=uniform_prediction, average="macro"),
                f1_score(y_true=y_test, y_pred=distance_prediction, average="macro"))
def generate_final_predict_df(input_prediction):
    input_prediction = input_prediction.astype(int)
    X_predict["Target"] = input_prediction

    tmp_household_predict_df = X_predict.set_index("idhogar")
    tmp_X_predict = all_predict_df.loc[all_predict_df["parentesco1"] == 1, :].set_index("idhogar")
    tmp_X_predict.loc[tmp_household_predict_df.index, "Target"] = tmp_household_predict_df.Target
    tmp_X_predict = tmp_X_predict.set_index("Id")
    tmp_all_predict_df = all_predict_df.set_index("Id")
    tmp_all_predict_df.loc[tmp_X_predict.index, "Target"] = tmp_X_predict.Target
    tmp_all_predict_df["Target"] = tmp_all_predict_df["Target"].fillna(1).astype(int)
    tmp_all_predict_df

    tmp_all_predict_df["Target"] = tmp_all_predict_df["Target"].fillna(1).astype(int)
    final_predict_df = tmp_all_predict_df.reset_index()[["Id", "Target"]]
    return final_predict_df

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
def plot_train_test(x, y, data, hue, title=None, x_label=None, y_label=None):
    
    # First check if y is a list
    y_is_list = True if isinstance(y, list) else False
    
    x_data_list = []
    y_data_list = []
    
    # Check if the hue is specified
    hue_list = data.loc[:, hue].unique()
    for cat in hue_list:
        x_data = data[data[hue] == cat][x]
        x_data_list.append(x_data)
        y_data = data[data[hue] == cat][y]
        y_data_list.append(y_data)

    data = []
    style_list = [None, 'dash', 'dot']
    colour_list = ['rgb(22, 96, 167)', 'rgb(205, 12, 24)']
    for x_data, y_data, cat, colour in zip(x_data_list, y_data_list, hue_list, colour_list):
        if(isinstance(y_data, pd.Series)):
            trace = go.Scatter(x=x_data, y=y_data, mode='lines+markers', name=cat )
            data.append(trace)
        elif(isinstance(y_data, pd.DataFrame)):
            # Means that y is a list of column names
            for col, style in zip(y, style_list):
                y_series = y_data[col]
                trace = go.Scatter(x=x_data, y=y_series, name= '+'.join([str(cat), col]),
                                   line=dict(
                                       dash = style,
                                       color=colour,
                                   )
                                  )
                data.append(trace)
    
    if x_label == None:
        x_label = x
    if y_label == None:
        if y_is_list:
            y_label = '+'.join(y)
        else:
            y_label = y
            
    layout = dict(
        title=title,
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label)
    )
    
    fig = dict(data=data, layout=layout)
    iplot(fig)
from math import ceil
def subplot_test_train(x, y, z, data,hue=None, title=None, x_label=None, y_label=None):
    
    y_is_list = True if isinstance(y, list) else False
    
    if x_label == None:
        x_label = x
    if y_label == None:
        if y_is_list:
            y_label = 'f1_macro score'
        else:
            y_label = y
    
    subplot_val_list = data.loc[:, z].unique()
    subplot_num = len(subplot_val_list)
    
    subplot_titles=[]
    for val in subplot_val_list:
        subplot_titles.append(': '.join([z, str(val)]))
    
    fig = tools.make_subplots(cols=2, rows=ceil(subplot_num/2), subplot_titles=subplot_titles, print_grid=False)
    
    original_data = data
    for i, val in enumerate(subplot_val_list):
        
        data = original_data[original_data[z] == val]
        xaxis_index = 'xaxis' + str(i + 1)
        yaxis_index = 'yaxis' + str(i + 1)
        
        x_data_list = []
        y_data_list = []
        
        if hue != None:
            hue_list = data.loc[:, hue].unique()
            for cat in hue_list:
                x_data = data[data[hue] == cat][x]
                x_data_list.append(x_data)
                y_data = data[data[hue] == cat][y]
                y_data_list.append(y_data)     
        else:
            hue_list = []
    
        style_list = [None, 'dash', 'dot']
        colour_list = ['rgb(22, 96, 167)', 'rgb(205, 12, 24)', 'rgb(12, 205, 24)' ]

        for x_data, y_data, cat, colour in zip(x_data_list, y_data_list, hue_list, colour_list):
            if(isinstance(y_data, pd.Series)):
                trace = go.Scatter(x=x_data, y=y_data, mode='lines+markers' )
                fig.append_trace(trace, i//2 + 1, (i%2 + 1))
            elif(isinstance(y_data, pd.DataFrame)):
                # Means that y is a list of column names
                for col, style in zip(y, style_list):
                    y_series = y_data[col]
                    trace = go.Scatter(x=x_data, y=y_series, name= '+'.join([cat, col]),
                                       line=dict(
                                           dash=style,
                                           color=colour,
                                       )
                                      )
                    fig.append_trace(trace, i//2 + 1, (i%2 + 1))
                    if i == subplot_num - 1 or i == subplot_num - 2:
                        fig['layout'][xaxis_index].update(title = x_label)
                    if i%2 == 0:
                        fig['layout'][yaxis_index].update(title = y_label)

    fig['layout'].update(title='SVC Optimised Parameters Searching')
    iplot(fig)
#     for val in subplot_val_list:
        
def ada_report(ada_models, X_train, y_train, X_test, y_test):
    
    ada_tree_real, ada_tree_discrete = ada_models
    
    real_test_f1 = []
    real_train_f1 = []
    discrete_test_f1 = []
    discrete_train_f1 = []
    
    trees_discrete_num = len(ada_tree_discrete)
    trees_real_num = len(ada_tree_real)
    
    for real_test_predict, real_train_predict, discrete_test_predict, discrete_train_predict in zip(
        ada_tree_real.staged_predict(X_test),
        ada_tree_real.staged_predict(X_train),
        ada_tree_discrete.staged_predict(X_test),
        ada_tree_discrete.staged_predict(X_train)
    ):
        real_test_f1.append(
            f1_score(y_pred=real_test_predict, y_true=y_test, average='macro')
        )
        
        real_train_f1.append(
            f1_score(y_pred=real_train_predict, y_true=y_train, average='macro')
        )
        
        discrete_test_f1.append(
            f1_score(y_pred=discrete_test_predict, y_true=y_test, average='macro')
        )
        
        discrete_train_f1.append(
             f1_score(y_pred=discrete_train_predict, y_true=y_train, average='macro')
        )
        
    f1s = [
        real_test_f1,
        real_train_f1,
        discrete_test_f1,
        discrete_train_f1, 
    ]
    labels = ["SAMME.R: Test", "SAMME.R: Train", "SAMME: Test", "SAMME: Train"]
    colours = ['rgb(22, 96, 167)', 'rgb(22, 96, 167)', 'rgb(205, 12, 24)', 'rgb(205, 12, 24)']
    styles = [None, "dash", None, "dash"]
        
    fig = tools.make_subplots(cols=1, rows=3, print_grid=False, shared_xaxes=True)
    
    for f1, label, colour, style in zip(f1s, labels, colours, styles):
        trace = go.Scatter(
            x=np.arange(1, trees_real_num, 1),
            y=f1,
            name= label,
            line=dict(
                color=colour,
                dash=style
            )
        )
        
        fig.append_trace(trace, 1, 1)
    
    fig["layout"]["yaxis1"].update(title="F1 Macro Score")
    
    trace_real_error = go.Scatter(
        x=np.arange(1, trees_real_num, 1),
        y=ada_tree_real.estimator_errors_,
        name="SAMME.R: Error",
#         line=dict(
#             color='rgb(22, 96, 167)',
#             width=3
#         )
    )
    
    trace_discrete_error = go.Scatter(
        x=np.arange(1, trees_real_num, 1),
        y=ada_tree_discrete.estimator_errors_,
        name="SAMME: Error",
#         line=dict(
#             color='rgb(205, 12, 24)',
#             width=3
#         )
    )
    
    fig.append_trace(trace_real_error, 2, 1)
    fig.append_trace(trace_discrete_error, 2, 1)
    fig["layout"]["yaxis2"].update(title="Error")
    
    trace_real_weights = go.Scatter(
        x=np.arange(1, trees_real_num, 1),
        y=ada_tree_real.estimator_weights_,
        name="SAMME.R: Weights",
    )
    
    trace_discrete_weights = go.Scatter(
        x=np.arange(1, trees_real_num, 1),
        y=ada_tree_discrete.estimator_weights_,
        name="SAMME: Weights",
    )
    
    fig.append_trace(trace_real_weights, 3, 1)
    fig.append_trace(trace_discrete_weights, 3, 1)
    fig["layout"]["yaxis3"].update(title="Weights")
    
    
    fig["layout"].update(title="Adaboost Discrete and Real Algorithm Comparison")
    iplot(fig)
def plot_features(feature_importances, index, most_important=True):
    indices = np.argsort(feature_importances)[::-1]
    indices = indices[:index]

    # Visualise these with a barplot
    plt.subplots(figsize=(20, 15))
    g = sns.barplot(y=X.iloc[:, 3:].columns[indices], x = lgb_clf.feature_importances_[indices], orient='h')
    g.set_xlabel("Relative importance",fontsize=20)
    g.set_ylabel("Features",fontsize=20)
    g.tick_params(labelsize=15)
    g.set_title("LightGBM feature importance", fontsize=20);
def plot_submission(df):
    trace = go.Scatter(x=df.index, y=df["public score"], mode='lines+markers', text=df["Note"])
    
    annotations_list = []
    for index, row in df.iterrows():
        annotations_list.append(
            dict(
                x=index,
                y=row[2],
                xref='x',
                yref='y',
                text=row[0],
                showarrow=True,
                arrowhead=7
            )
        )
        
    layout = go.Layout(annotations=annotations_list, title="Hover Over The Points To See My Note",
                       xaxis=dict(title="Submission Sequence"),
                       yaxis=dict(title="Finial F1 Score")
                      )
    
    fig = go.Figure(data=[trace], layout=layout)

    iplot(fig)
poverty_map = {
    1:"extreme poverty", 
2 : "moderate poverty" ,
3 : "vulnerable households", 
4 : "non vulnerable households"
}
raw_train_df = pd.read_csv(prefix + "train.csv")
raw_predict_df = pd.read_csv(prefix + "test.csv")
example_predict_df = pd.read_csv(prefix + "sample_submission.csv")
raw_predict_df = pd.concat([raw_predict_df, pd.Series(np.nan, name='Target')], axis=1)
raw_all_df = pd.concat([raw_train_df, raw_predict_df], ignore_index=True)
raw_all_df["edjefe"] = raw_all_df["edjefe"].apply(lambda x: 1 if x == "yes" else x).apply(lambda x: 0 if x == "no" else x)

raw_all_df["edjefa"] = raw_all_df["edjefa"].apply(lambda x: 1 if x == "yes" else x).apply(lambda x: 0 if x == "no" else x)

raw_all_df["dependency"] = raw_all_df["dependency"].apply(lambda x: 1 if x == "yes" else x).apply(lambda x: 0 if x == "no" else x)

raw_all_df["edjefe"] = pd.to_numeric(raw_all_df["edjefe"])

raw_all_df["edjefa"] = pd.to_numeric(raw_all_df["edjefa"])

raw_all_df["dependency"] = pd.to_numeric(raw_all_df["dependency"])
raw_all_df.isnull().sum()[raw_all_df.isnull().sum() > 0]
clean_all_df = raw_all_df.drop(["v2a1", "rez_esc"], axis=1)
clean_all_df["v18q1"].fillna(0, inplace=True)
clean_all_df.isnull().sum()[clean_all_df.isnull().sum() > 0]
id_col = ['Id', 'idhogar', 'Target']
ind_bool_col = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone',]

ind_ordered_col = ['escolari', 'age']
hh_bool_col = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area2' , 'area1'] # redundant

hh_ordered_col = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
               'v18q1', 'hogar_nin', 'tamhog','tamviv','hhsize', 'r4t3', 'hogar_total',# redundant
              'hogar_adul','hogar_mayor',  'bedrooms', 'qmobilephone']

hh_cont_col = ['dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
sqr_col = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
clean_all_df.drop(sqr_col, axis=1, inplace=True)
cols_list = id_col + ind_bool_col + ind_ordered_col + hh_bool_col + hh_ordered_col + hh_cont_col
clean_all_df = clean_all_df[cols_list]
clean_all_df.isnull().sum()[clean_all_df.isnull().sum() > 0]
temp_all_df = pd.concat([clean_all_df, pd.Series(np.argmax(clean_all_df.loc[:, "instlevel1" : "instlevel9"].values, axis=1), name="eduindex")], axis=1)
edu_median_dict = temp_all_df.groupby("eduindex")["meaneduc"].median().to_dict()
def fill_na(x):
    if pd.isnull(x["meaneduc"]):
        x.loc["meaneduc"] = edu_median_dict[x.loc["eduindex"]]
    return x
temp_all_df = temp_all_df.apply(fill_na, axis=1)
# edu_list = clean_all_df.loc[:, "instlevel1" : "instlevel9"].columns
clean_all_df = temp_all_df.drop("eduindex", axis=1)
heads_all_df = clean_all_df[clean_all_df.parentesco1 == 1]
corr_matrix =  clean_all_df[clean_all_df.Target.notnull()].corr().abs()
np.fill_diagonal(corr_matrix.values, np.NaN)
sorted_corr_series = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)
sorted_corr_series[sorted_corr_series > 0.90].head(3)
rm_cols = ["tamhog", "r4t3", "hhsize", "hogar_total", "male", "area1"]
clean_all_df.drop(rm_cols, axis=1, inplace=True)
for col in rm_cols:
    for _list in [ind_bool_col, ind_ordered_col, hh_bool_col, hh_cont_col, hh_ordered_col]:
        if col in _list:
            _list.remove(col)
roof_list = ["techozinc", "techoentrepiso", "techocane", "techootro", "cielorazo"]

clean_all_df["cielorazo"] = clean_all_df[roof_list[:-1]].sum(axis=1).map({1:0, 0:1})
no_head_household_ids = clean_all_df.groupby("idhogar").sum()[clean_all_df.groupby("idhogar").sum()["parentesco1"] == 0].index
clean_drop_no_head_df = clean_all_df.copy()
for x in no_head_household_ids:
    clean_drop_no_head_df.drop(clean_drop_no_head_df.loc[clean_drop_no_head_df.idhogar == x, :].index, axis=0, inplace=True)
clean_heads_df = clean_drop_no_head_df.loc[clean_drop_no_head_df.parentesco1 == 1]

heads_dict = clean_heads_df[["idhogar", "Target"]].set_index("idhogar").to_dict()["Target"]
def synchronise_diff_target(x):
    x.Target = heads_dict[x.idhogar]
    return x

clean_drop_no_head_df = clean_drop_no_head_df.apply(synchronise_diff_target, axis=1)
aggregation_list = ["max", "sum", "min", "mean", "std"]
hh_agg_features = clean_drop_no_head_df[["idhogar"] + ind_bool_col + ind_ordered_col].groupby("idhogar").agg(aggregation_list)
hh_agg_features.columns = ["_".join(x) for x in hh_agg_features.columns.ravel()]

hh_agg_features = hh_agg_features.fillna(0)
clean_heads_df = clean_drop_no_head_df[clean_drop_no_head_df.parentesco1 == 1].loc[:, id_col + hh_bool_col + hh_cont_col + hh_ordered_col].set_index("idhogar")

clean_heads_df = pd.concat([clean_heads_df, hh_agg_features], axis=1, sort=False).reset_index().rename(columns={"index":'idhogar'})
X = clean_heads_df[clean_heads_df.Target.notna()].reset_index(drop=True)
y = X.Target
X_predict = clean_heads_df[clean_heads_df.Target.isna()].reset_index(drop=True)
all_predict_df = clean_all_df[clean_all_df.Target.isnull()]
plot_hist(X, "age_mean", title="Average Age In Same Household" ,x_label="Average age")
plot_hist(X, "escolari_mean", x_label="Average years of schooling", title="Average Years of Schooling In Same Household")
plot_hist(X, "hacdor", x_label="Overcrowding by bedrooms", title="If the Bedrooms are Overcrowded")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X.iloc[:, 3:], X_predict.iloc[:, 3:] = scaler.fit_transform(X.iloc[:, 3:]), scaler.fit_transform(X_predict.iloc[:, 3:])
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
numeric_train_df = X.iloc[:, 3:]
label_series =  X.Target.reset_index(drop=True)
pca = PCA(n_components=3, whiten=True, svd_solver='auto', random_state=166446054) 
pca_train_array = pca.fit_transform(numeric_train_df)
pca_train_df = pd.DataFrame(pca_train_array)

pca_train_df = pd.concat([pca_train_df, label_series], axis=1)
pca_3d_plot(pca_train_df=pca_train_df)
from sklearn.cluster import KMeans
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.pipeline import Pipeline
result_dict = search_optimised_k([4, 8, 16, 32, 64, 128, 259], random_state=16446054, X=X.iloc[:, 3:])
km_param_results = pd.DataFrame(result_dict)
ax = km_param_results.set_index("n_clusters").plot(figsize=(12,8), fontsize=15, title="The Silhouette Score Relationship With Number of Centroid")
ax.set_ylabel("Silhouette Score")
km_predict = KMeans(n_jobs=-1, n_clusters=4, random_state=16446054).fit_predict(X.iloc[:, 3:])
km_X = pd.concat([pd.Series(km_predict, name="k_means_predict").map({0:"centroid1",
                                                                     1:"centroid2",
                                                                     2: "centroid3",
                                                                     3: "centroid4"
                                                                    }), X.iloc[:, 2:]], axis=1)

plot_hist(km_X, "k_means_predict", x_label="KMeans Prediction", title="K Means Clustering Prediction")
result_dict = search_optimised_k([4, 8, 16, 32, 64, 128, 259], random_state=16446054, X=pca_train_df)
km_param_results = pd.DataFrame(result_dict)
ax = km_param_results.set_index("n_clusters").plot(figsize=(12,8), fontsize=15, title="The Silhouette Score Relationship With Number of Centroid")
ax.set_ylabel("Silhouette Score")
km_predict = KMeans(n_jobs=-1, n_clusters=4, random_state=16446054).fit_predict(pca_train_df)
km_X = pd.concat([pd.Series(km_predict, name="k_means_predict").map({0:"centroid1",
                                                                     1:"centroid2",
                                                                     2: "centroid3",
                                                                     3: "centroid4",
                                                                     4: "centroid5",
                                                                     5: "centroid6",
                                                                     6: "centroid7",
                                                                     7: "centroid8"
                                                                    }), X.iloc[:, 2:]], axis=1)

plot_hist(km_X, "k_means_predict", x_label="KMeans Prediction", title="K Means Clustering Prediction")
from sklearn.model_selection import train_test_split
my_knn_agent = knn_agent()
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, 3:], y, test_size=0.2, random_state=16446054, stratify=y)
my_knn_agent.optimise_params(X_train=X_train, y_train=y_train, X_dev=X_test, y_dev=y_test, max_k_num=50)
uniform_f1, distance_f1 = my_knn_agent.evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print("Unifrom test set f1 score: ", uniform_f1, " Distance test set f1 score: ", distance_f1)
uniform_prediction, distance_prediction = my_knn_agent.predict(X_train=X_train, y_train=y_train, X_predict=X_predict.iloc[:, 3:])
# final_predict_df = generate_final_predict_df(uniform_prediction)

# final_predict_df.to_csv("knn_submission.csv", index=False)
from sklearn.model_selection import GridSearchCV
cachedir = mkdtemp()

knn_pipe = Pipeline([('reduce_dim', PCA(random_state=16446054)), ('classify', my_knn_agent.uniform_knn)], memory=cachedir)
num_features_list = np.arange(10, 200, 20)
knn_param_grid = [
    {
        'reduce_dim': [PCA(random_state=16446054)],
        'reduce_dim__n_components': num_features_list,
    }, 
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': num_features_list,
    }
]

reducer_labels = ['PCA', 'KBest(chi2)']
knn_grid = GridSearchCV(estimator=knn_pipe, cv=5, n_jobs=4, param_grid=knn_param_grid, scoring='f1_macro')
_ = knn_grid.fit(X.iloc[:, 3:], y)
report(knn_grid.cv_results_)
knn_params_results = pd.DataFrame(knn_grid.cv_results_)
knn_params_results = pd.concat([knn_params_results.param_reduce_dim__n_components.fillna(knn_params_results.param_reduce_dim__k).rename("num_features"),
                                knn_params_results.drop(["param_reduce_dim__k", "param_reduce_dim__n_components"], axis=1)], axis=1)
knn_params_results.loc[:, "param_reduce_dim"] = knn_params_results.param_reduce_dim.apply(lambda x: "PCA" if str(x).startswith("PCA") else "SelectKBest").rename("algorithm")
plot_train_test("num_features", ["mean_test_score", "mean_train_score"], 
                hue="param_reduce_dim", data=knn_params_results, y_label="F1 Score",
                title="Average Train/Test F1 Score Relationship with the Number of Features")
# Clear the cache directory when you don't need it anymore
rmtree(cachedir)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
cachedir = mkdtemp()

svc_pipe = Pipeline([('reduce_dim', PCA(random_state=166446054)), ('classify', SVC(random_state=16446054, class_weight="balanced"))], memory=cachedir)
svc_param_grid = [
    {
        'reduce_dim__n_components': [10, 70, 140, 200],
        'classify__kernel': ['rbf', 'poly'],
        'classify__C': [0.1, 1, 10, 50],
    }
]
svc_grid = GridSearchCV(svc_pipe, cv=5, n_jobs=4, param_grid=svc_param_grid, scoring='f1_macro', refit=True)

_ = svc_grid.fit(X.iloc[:, 3:], y)
report(svc_grid.cv_results_)
svc_params_results = pd.DataFrame(svc_grid.cv_results_)
subplot_test_train(x="param_reduce_dim__n_components", y=["mean_test_score", "mean_train_score"], z="param_classify__C", data=svc_params_results, hue="param_classify__kernel")
rmtree(cachedir)
# clf.fit(X.iloc[:, 3:], X.Target)
# svc_test_predict = clf.predict(heads_test_df.iloc[:, 3:])
# print("SVC test prediction f1 score: ",f1_score(y_pred=svc_test_predict, y_true=heads_test_df.Target, average="macro"))
# svc_predict = clf.predict(X_predict.iloc[:, 3:])

# final_predict = generate_final_predict_df(svc_predict)

# final_predict.to_csv("svc_submission.csv", index=False)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
ada_tree_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5
)

ada_tree_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME"
)
ada_tree_real.fit(X_train, y_train)
ada_tree_discrete.fit(X_train, y_train)
ada_report(ada_models=(ada_tree_real, ada_tree_discrete), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
cachedir = mkdtemp()

ada_pipe = Pipeline([('reduce_dim', PCA(random_state=166446054)),
                     ('classify', AdaBoostClassifier(random_state=16446054))], memory=cachedir)
ada_param_grid = [
    {
        'reduce_dim__n_components': [10, 70, 140, 259],
        'classify__algorithm': ['SAMME.R', 'SAMME'],
        'classify__learning_rate': [1],
        'classify__n_estimators': [300],
        'classify__base_estimator': [DecisionTreeClassifier(max_depth=2)]
    }
]
ada_grid = GridSearchCV(ada_pipe, cv=5,
                        n_jobs=4, param_grid=ada_param_grid,
                        scoring='f1_macro', refit=True, )#verbose=20) # This can show the progress of searching

_ = ada_grid.fit(X.iloc[:, 3:], y)
report(ada_grid.cv_results_)
ada_param_results = pd.DataFrame(ada_grid.cv_results_)
plot_train_test(x="param_reduce_dim__n_components", y=["mean_test_score", "mean_train_score"], data=ada_param_results, hue="param_classify__algorithm")
# real_test_predict = np.array(ada_real_test_result).mean(axis=0).round().astype(int)
# discrete_test_predict = np.array(ada_discrete_test_result).mean(axis=0).round().astype(int)
# print("Real test prediction f1 score: ",f1_score(y_pred=real_test_predict, y_true=heads_dev_df.Target, average="macro"))
# print("Discrete test prediction f1 score: ",f1_score(y_pred=discrete_test_predict, y_true=heads_dev_df.Target, average="macro"))

# real_test_predict = ada_tree_real.predict(heads_test_df.iloc[:, 3:])
# discrete_test_predict = ada_tree_discrete.predict(heads_test_df.iloc[:, 3:])
# print("Real test prediction f1 score: ",f1_score(y_pred=real_test_predict, y_true=heads_test_df.Target, average="macro"))
# print("Discrete test prediction f1 score: ",f1_score(y_pred=discrete_test_predict, y_true=heads_test_df.Target, average="macro"))

# real_predict = ada_tree_real.predict(X_predict.iloc[:, 3:]).astype(int)
# discrete_predict = ada_tree_discrete.predict(X_predict.iloc[:, 3:]).astype(int)

# real_final_predict = generate_final_predict_df(real_predict)
# discrete_finial_predict = generate_final_predict_df(discrete_predict)

# real_final_predict.to_csv("real_ada_submission.csv", index=False)

# discrete_finial_predict.to_csv("discrete_ada_submission.csv", index=False)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
lgb_clf = lgb.LGBMClassifier(learning_rate=0.01, n_estimators=15000, 
                             objective='multiclass', matric='logloss', boosting_type='dart',
                             num_leaves=700, max_depth=10,
                             class_weight='balanced',  silent=True, n_jobs=-1,
                             colsample_bytree =  0.93, min_child_samples = 95,  subsample = 0.96)
kfold = 10
kf = StratifiedKFold(n_splits=kfold, shuffle=True)

predicts_result = []
test_result = []
for train_index, test_index in kf.split(X.iloc[:, 3:], X.Target):
    print("#"*10)
    X_train, X_val = X.iloc[:, 3:].iloc[train_index], X.iloc[:, 3:].iloc[test_index]
    y_train, y_val = X.Target.iloc[train_index], X.Target.iloc[test_index]
    lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=1000, verbose=10000) #eval_metric=f1_macro_evaluation)
    test_result.append(f1_score(y_pred=lgb_clf.predict(X_val), y_true=y_val, average="macro"))
    predicts_result.append(lgb_clf.predict(X_predict.iloc[:, 3:]))
for result in test_result:
    print(result)
print("Mean test f1: ", np.mean(test_result))
lgb_train_predict = lgb_clf.predict(X.iloc[:, 3:])
lgb_test_f1 = f1_score(y_pred=lgb_train_predict, y_true=X.Target, average="macro")
print("LGBM train f1: ", lgb_test_f1)
# lgb_predict = np.array(predicts_result).mean(axis=0).round().astype(int)
# lgb_predict = lgb_clf.predict(X_predict.iloc[:, 3:])

# final_predict = generate_final_predict_df(lgb_predict)

# final_predict.to_csv('submission.csv', index = False)
plot_features(lgb_clf.feature_importances_, index=20)
submission_history = pd.read_excel("../input/competitionjournal/costa_rican_competition_result.xlsx")

submission_history.set_index("sequence", inplace=True)

plot_submission(submission_history)
