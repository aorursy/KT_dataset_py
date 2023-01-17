!pip install pycaret

from pycaret.utils import version
version()
from pycaret.datasets import get_data
from pycaret.classification import *
diabetes = get_data('diabetes')
clf = setup(diabetes, target ='Class variable', silent=True, sampling=False)
compare_models()
cb1 = create_model('catboost')
tuned_cb = tune_model(cb1, optimize="AUC", n_iter=50)
interpret_model(tuned_cb)
predictions=predict_model(tuned_cb)
finalize_model(tuned_cb)