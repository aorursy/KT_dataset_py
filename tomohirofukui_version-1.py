import pandas as pd

import pandas_profiling as pdp
import pandas as pd

application_test = pd.read_csv("../input/home-credit-default-risk/application_test.csv")

application_train = pd.read_csv("../input/home-credit-default-risk/application_train.csv")

bureau_balance = pd.read_csv("../input/home-credit-default-risk/bureau_balance.csv")

bureau = pd.read_csv("../input/home-credit-default-risk/bureau.csv")

credit_card_balance = pd.read_csv("../input/home-credit-default-risk/credit_card_balance.csv")

# HomeCredit_columns_description = pd.read_csv("../input/home-credit-default-risk/HomeCredit_columns_description.csv")

installments_payments = pd.read_csv("../input/home-credit-default-risk/installments_payments.csv")

POS_CASH_balance = pd.read_csv("../input/home-credit-default-risk/POS_CASH_balance.csv")

previous_application = pd.read_csv("../input/home-credit-default-risk/previous_application.csv")

sample_submission = pd.read_csv("../input/home-credit-default-risk/sample_submission.csv")
application_columns = application_train.columns
len(application_columns)
len(application_train)
pdp.ProfileReport(application_train[application_columns[1:9]], missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[9:20]], missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[20:30]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[30:40]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[40:50]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[50:60]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[60:70]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[70:80]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[80:90]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[90:100]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[100:110]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(application_train[application_columns[110:]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
bureau_balance_columns = bureau_balance.columns
len(bureau_balance_columns)
bureau_balance.head()
bureau_balance[bureau_balance['SK_ID_BUREAU'] == 5715448]
pdp.ProfileReport(bureau_balance.sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
bureau_columns = bureau.columns
len(bureau_columns)
bureau.head()
bureau[bureau['SK_ID_CURR'] == 215354]
pdp.ProfileReport(bureau[bureau_columns[:10]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})
pdp.ProfileReport(bureau[bureau_columns[10:]].sample(n=50000), missing_diagrams={

    'bar': False,

    'matrix': False,

    'heatmap': False,

    'dendrogram': False},

correlations={

    'pearson': False,

    'spearman': False,

    'kendall': False,

    'phi_k': False,

    'cramers': False,

    'recoded': False})