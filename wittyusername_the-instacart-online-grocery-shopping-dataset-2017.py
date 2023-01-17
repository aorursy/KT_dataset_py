import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import Image, display
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz
!apt-get install -y tree
%cd ../input
!tree
ITEMS = 10_000

# These are completely, entirely arbitrary. I generated them using:
#
# >>> import random
# >>> random.randint(1, 2**32 - 1)

RANDOM_STATE_SAMPLE_ONE =  596497102
RANDOM_STATE_SAMPLE_TWO = 1539131531
RANDOM_STATE_DEC_TREE   = 2343470337
products = pd.read_csv("products.csv")
products = products.sample(n=ITEMS, random_state=RANDOM_STATE_SAMPLE_ONE)
products = products.drop(columns=["product_id", "department_id"], axis=1)

products.head()
aisles = pd.read_csv("aisles.csv")
aisles.head()
# This is going to be our prediction target
y = products.aisle_id

_products = pd.get_dummies(products)
def id_to_aisle(_id):
    return aisles.loc[aisles.aisle_id == _id].aisle
# Define model. Specify a number for random_state to ensure same results each run
products_model = DecisionTreeRegressor(random_state=RANDOM_STATE_DEC_TREE)

# Fit model
products_model.fit(_products, y)
predictions = products_model.predict(_products.head())

for idx, prediction in enumerate(predictions):
    print(products.iloc[idx].product_name, id_to_aisle(prediction))
products = pd.read_csv("products.csv")
products = products.sample(n=ITEMS, random_state=RANDOM_STATE_SAMPLE_TWO)
products = products.drop(columns=["product_id", "department_id"], axis=1)
products.head()
_products = pd.get_dummies(products)

predictions = products_model.predict(_products)

n_correct = 0

for idx, prediction in enumerate(predictions):
    if products.iloc[idx].aisle_id == prediction:
        n_correct += 1

print(f"Got {n_correct} products correct, out of {len(products)} (accuracy: {round((n_correct/len(products)) * 100, 2)}%).")
products = pd.read_csv("products.csv")
products_f = products.sample(n=ITEMS, random_state=RANDOM_STATE_SAMPLE_ONE)
products_l = products.sample(n=ITEMS, random_state=RANDOM_STATE_SAMPLE_TWO)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))

pd.value_counts(products_f.aisle_id).plot.hist(ax=ax1)
pd.value_counts(products_l.aisle_id).plot.hist(ax=ax2)
export_graphviz(products_model, "../model.dot")
!dot -Tpng ../model.dot -o ../model.png
display(Image(filename="../model.png"))
!rm ../model.png