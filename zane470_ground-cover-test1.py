# Get the newest version of tensorflow and Ludwig 



!pip install https://github.com/uber/ludwig/archive/master.zip
# Train a model using our RGB training data



import pandas as pd



training_dataframe = pd.read_csv('../input/ground-cover-dataset-1/rgb_training_data.csv')

#training_dataframe[['red','green','blue']] = training_dataframe[['red','green','blue']].div(training_dataframe['clear'].values,axis=0)



model_definition = {

    'input_features': [

        {'name': 'clear', 'type': 'numerical'},

        {'name': 'lux', 'type': 'numerical'},

        {'name': 'colorTemp', 'type': 'numerical'},

        {'name': 'red', 'type': 'numerical'},

        {'name': 'green', 'type': 'numerical'},

        {'name': 'blue', 'type': 'numerical'}

    ],

    'output_features': [{'name': 'background', 'type': 'category'}],

    'training': {'epochs': 250}

}



from ludwig.api import LudwigModel



model = LudwigModel(model_definition)

train_stats = model.train(training_dataframe)



from ludwig import visualize

visualize.learning_curves([train_stats],None)
# Use the model to determine ground cover of future data



predict_dataframe = pd.read_csv('../input/ground-cover-dataset-1/rgb_predict_data.csv')

#predict_dataframe[['red','green','blue']] = predict_dataframe[['red','green','blue']].div(predict_dataframe['clear'].values,axis=0)



predictions = model.predict(predict_dataframe)

print(predictions)



model.close()