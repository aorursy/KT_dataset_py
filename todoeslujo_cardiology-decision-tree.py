import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
df=pd.read_excel("../input/CNumerical.xls", "Sheet1")
print(df.head(10))
# extracting only age, sex,chest pain type,blood pressure,cholesterol,maximum heart rate
df2 = df[['age','sex','chest pain type', 'blood pressure', 'cholesterol','maximum heart rate']]
df2.head(3)
def create_label_encoder_dict(df2):
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder_dict = {}
    for column in df2.columns:
        # Only create encoder for categorical data types
        if not np.issubdtype(df2[column].dtype, np.number) and column != 'age':
            label_encoder_dict[column]= LabelEncoder().fit(df2[column])
    return label_encoder_dict

label_encoders = create_label_encoder_dict(df2)
print("Encoded Values for each Label")
print("="*32)
for column in label_encoders:
    print("="*32)
    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))
    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)
# Apply each encoder to the data set to obtain transformed values
df3 = df2.copy() # create copy of initial data set
for column in df3.columns:
    if column in label_encoders:
        df3[column] = label_encoders[column].transform(df3[column])

print("Transformed data set")
print("="*32)
df3.head(10)
# separate our data into dependent (Y) and independent(X) variables
X_data = df3[['blood pressure','sex','age', 'maximum heart rate']]
Y_data = df3['chest pain type']
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier(max_depth=10, criterion='entropy') # Change the max_depth to 10 or another number
# build classifier
clf.fit(X_data, Y_data)

pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100) ], index = X_data.columns, columns = ['Feature Significance in Decision Tree'])
#prediction
prediction=clf.predict([[130,1,45,200]])
prediction


from IPython.display import Image  
from sklearn.tree import export_graphviz 
import graphviz
#import pydotplus
dot_data = tree.export_graphviz(clf,
                                feature_names=X_data.columns,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
 
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)
 
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
 
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
graph.write_png('tree.png')
graph = graph_from_dot_data(dot_data)  
graph
