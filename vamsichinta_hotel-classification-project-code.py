import nltk
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1 = pd.read_excel('../input/Tripadvisor_Dataset.xlsx')
print(df1.shape)
df1 = df1.dropna(subset=['Review Text'])
print(df1.shape)
df1.to_excel('Tripadvisor_Dataset_Filled.xlsx', index=False)
df1 = pd.read_excel('Tripadvisor_Dataset_Filled.xlsx')
df1.describe()
df1.groupby('Review Rating').describe()
authentic_df = df1[df1['Likes on Review'] > 1]
df_review = list(authentic_df['Review Text'].values)
scores_vadar = []
scores_textblob = []

sid = SentimentIntensityAnalyzer()
for review in df_review:
    ss = sid.polarity_scores(str(review))
    tb = TextBlob(str(review))
    snt = tb.sentiment.polarity
    scores_vadar.append((str(review), ss['compound']))
    scores_textblob.append((str(review), snt))
authentic_df_scores = pd.DataFrame(scores_vadar, columns=['Review Text', 'Score_Vadar'])
scores_tb = pd.DataFrame(scores_textblob, columns=['Review Text', 'Score_Textblob'])

# authentic_df_scores.to_csv('Sentiment_scores_authentic.csv', index=False)
left = authentic_df.set_index('Review Text')
right = authentic_df_scores.set_index('Review Text')
result = left.join(right)

another = scores_tb.set_index('Review Text')

result = result.join(another)
result = result.dropna(subset=['Score_Vadar'])
result = result.dropna(subset=['Score_Textblob'])
result['Vador_Sentiment'] = ''
for i, row in result.iterrows():
    if row['Score_Vadar'] > 0:
        result.loc[i,'Vador_Sentiment'] = 'Good'
    elif row['Score_Vadar'] == 0:
        result.loc[i,'Vador_Sentiment'] = 'Neutral'
    elif row['Score_Vadar'] < 0:
        result.loc[i,'Vador_Sentiment'] = 'Bad'
result['Textblob_Sentiment'] = ''
for i, row in result.iterrows():
    if row['Score_Textblob'] > 0:
        result.loc[i,'Textblob_Sentiment'] = 'Good'
    elif row['Score_Textblob'] == 0:
        result.loc[i,'Textblob_Sentiment'] = 'Neutral'
    elif row['Score_Textblob'] < 0:
        result.loc[i,'Textblob_Sentiment'] = 'Bad'
result['Actual_Sentiment'] = ''
for i, row in result.iterrows():
    if row['Review Rating'] in [3,4,5]:
        result.loc[i,'Actual_Sentiment'] = 'Good'
    elif row['Review Rating'] == 2:
        result.loc[i,'Actual_Sentiment'] = 'Noise'
    else:
        result.loc[i,'Actual_Sentiment'] = 'Bad'
result.to_excel('Dataset_with_actual_and_predicted_labels.xlsx')
vadar_correct_predictions = 0
textblob_correct_predictions = 0
total_predictions = 0
for i, row in result.iterrows():
    if row['Actual_Sentiment'] == 'Good' and row['Textblob_Sentiment'] == 'Good':
        textblob_correct_predictions += 1
    elif row['Actual_Sentiment'] == 'Good' and row['Textblob_Sentiment'] == 'Good':
        textblob_correct_predictions += 1
        
    if row['Actual_Sentiment'] == 'Good' and row['Vador_Sentiment'] == 'Good':
        vadar_correct_predictions += 1
    elif row['Actual_Sentiment'] == 'Bad' and row['Vador_Sentiment'] == 'Bad':
        vadar_correct_predictions += 1
        
    if row['Actual_Sentiment'] == 'Good' or row['Actual_Sentiment'] == 'Bad':
        total_predictions += 1

textblob_accuracy = (textblob_correct_predictions/total_predictions)*100
print('Textblob Accuracy: ', textblob_accuracy)
vadar_accuracy = (vadar_correct_predictions/total_predictions)*100
print('Vader Accuracy: ', vadar_accuracy)
temp_df = result.groupby('Name').describe()
temp_df

temp_df.to_csv('Data Quality Report.csv')
#precision = tp/tp+fp where fp is comments incorrectly classified as 'Good'
#recall = tp/tp+fn where fn is comments incorrectly classified as not 'Good'(Bad)
true_positives = 0
false_positives = 0
false_negatives = 0

for i, row in result.iterrows():
    if row['Actual_Sentiment'] == 'Good' and row['Vador_Sentiment'] == 'Good':
        true_positives += 1
    if row['Actual_Sentiment'] == 'Bad' and row['Vador_Sentiment'] == 'Good':
        false_positives += 1
    if row['Actual_Sentiment'] == 'Good' and row['Vador_Sentiment'] == 'Bad':
        false_negatives += 1

precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives+false_negatives)

F_score = 2*((precision*recall)/(precision+recall))

print('Evaluation metrics for Vader:')
print('Precision: ', round(precision, 3))
print('Recall: ', round(recall, 3))
print('F1 score(Harmonic Mean): ', round(F_score, 3))
#precision = tp/tp+fp where fp is comments incorrectly classified as 'Good'
#recall = tp/tp+fn where fn is comments incorrectly classified as not 'Good'(Bad)
true_positives = 0
false_positives = 0
false_negatives = 0

for i, row in result.iterrows():
    if row['Actual_Sentiment'] == 'Good' and row['Textblob_Sentiment'] == 'Good':
        true_positives += 1
    if row['Actual_Sentiment'] == 'Bad' and row['Textblob_Sentiment'] == 'Good':
        false_positives += 1
    if row['Actual_Sentiment'] == 'Good' and row['Textblob_Sentiment'] == 'Bad':
        false_negatives += 1

precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives+false_negatives)

F_score = 2*((precision*recall)/(precision+recall))

print('Evaluation metrics for Textblob:')
print('Precision: ', round(precision, 3))
print('Recall: ', round(recall, 3))
print('F1 score(Harmonic Mean): ', round(F_score, 3))
F_score = 2*((precision*recall)/(precision+recall))
F_score
