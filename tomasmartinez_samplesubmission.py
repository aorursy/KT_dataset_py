import pandas as pd 

# These lines were used to generate the sampleSubmision
ids = [i+1 for i in range(1526)]
predicted = [random.random() for e in ids ]

sampleSubmission = pd.DataFrame({'Id':ids,'Predicted':predicted})
sampleSubmission.to_csv('sampleSubmision.csv',index=False)
