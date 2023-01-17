from IPython.display import display, HTML

import pandas as pd

pd.set_option("display.max_columns", 1000)

pd.set_option("display.max_rows", 1000)

pd.set_option("display.min_rows", 1000)

pd.set_option("display.max_colwidth", 1000)

pd.set_option("display.expand_frame_repr", True)

df = pd.read_csv("/kaggle/input/covid19-concept-cluster-qontology/covid19_ontology_light_v0_2.txt", \

                 sep="\t", header=None)

cate_dict = {}

for i, row in df.iterrows():

    if not row[1] in cate_dict:

        cate_dict[row[1]] = []

        for word in row[0].split(","):

            cate_dict[row[1]].append("<code><font size= 3>" + word+ "</font></code>")

    else:

        for word in row[0].split(","):

            cate_dict[row[1]].append("<code><font size= 3>" + word+ "</font></code>")

for k, v in cate_dict.items():

    cate_dict[k] = ", ".join(v)

result_df = pd.DataFrame.from_dict(cate_dict, columns=["words"], orient="index")

result_df.style.set_properties(**{"text-align": "left"})