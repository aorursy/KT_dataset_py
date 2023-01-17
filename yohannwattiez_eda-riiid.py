import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
NROWS=10**7
dtype = {'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 

         'content_type_id': 'int8','task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 

         'prior_question_elapsed_time': 'float32','prior_question_had_explanation': 'boolean',

        }



nrows = 10**7

train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', chunksize=nrows, dtype=dtype)
j=0

for i in train:

    print(i.info(null_counts=True))

    j+=1

j-=1
print('Number of rows: %f' % (j*nrows+len(i)))
train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', nrows=NROWS, dtype=dtype)

train = train.dropna()

train = train[train['answered_correctly']!=-1]

train
train = train[train['content_type_id']==0]

train
train.describe()
print('Proportion of questions:%f' % (train.count().row_id/NROWS))

print('Proportion of unique questions:%f' % (len(train.content_id.unique())/len(train)))
train.timestamp.hist(bins=10)
train.timestamp.hist(bins=100)
train.iloc[:10**5].groupby('user_id').timestamp.hist(bins=10)
train.iloc[:10**5].groupby('user_id').timestamp.hist(bins=100)
print('Average number of question per student: %f' % train.groupby('user_id').count().mean().row_id)

print('Standard deviation of number of question per student: %f' % train.groupby('user_id').count().std().row_id)
print(train.groupby('user_id').count().describe().row_id)
train.groupby('user_id').count().row_id.hist(bins=10)
train.groupby('user_id').count().row_id.hist(bins=100)
print('Proportion of unique task_container_id: %f' % (len(train.task_container_id.unique())/NROWS))
train.groupby('task_container_id').count().describe().row_id
print('Percentage of questions that appears only once: %f' % ((train.groupby('content_id').count().row_id==1).mean()))
print('Percentage of unique question: %f' % (len(train.groupby('content_id').count())/len(train)))
train.content_id.hist(bins=10)
train.content_id.hist(bins=100)
print(train.groupby('user_id').mean().prior_question_elapsed_time.mean())

print(train.groupby('user_id').mean().prior_question_elapsed_time.std())
train.prior_question_elapsed_time.hist(bins=10)
train.prior_question_elapsed_time.hist(bins=100)
train.iloc[:10**5].groupby('user_id').prior_question_elapsed_time.hist(bins=10)
train.iloc[:10**5].groupby('user_id').prior_question_elapsed_time.hist(bins=100)
print('Percentage of questions that had an explanation: \n%s' % (train.prior_question_had_explanation.value_counts()/NROWS))
print('Description of True prior_question_had_explanation per user: \n%s' % (train.groupby(['user_id', 'prior_question_had_explanation']).count().row_id.reset_index()[train.groupby(['user_id', 'prior_question_had_explanation']).count().row_id.reset_index().prior_question_had_explanation==True].describe().row_id))

print('Number of user without True value: %d' % (len(train.user_id.unique())-len((train.groupby(['user_id', 'prior_question_had_explanation']).count().row_id.reset_index()[train.groupby(['user_id', 'prior_question_had_explanation']).count().row_id.reset_index().prior_question_had_explanation==True]))))
print('Description of False prior_question_had_explanation per user: \n%s' % (train.groupby(['user_id', 'prior_question_had_explanation']).count().row_id.reset_index()[train.groupby(['user_id', 'prior_question_had_explanation']).count().row_id.reset_index().prior_question_had_explanation==False].describe().row_id))

print('Number of user without False value: %d' % (len(train.user_id.unique())-len((train.groupby(['user_id', 'prior_question_had_explanation']).count().row_id.reset_index()[train.groupby(['user_id', 'prior_question_had_explanation']).count().row_id.reset_index().prior_question_had_explanation==False]))))
print('Percentage of questions answered correctly: %f' % train['answered_correctly'].mean())

train['answered_correctly'].hist(bins=10)
count_answered_correctly_true_per_user = (train.groupby(['user_id', 'answered_correctly']).count().reset_index()[train.groupby(['user_id', 'answered_correctly']).count().reset_index().answered_correctly==1].set_index('user_id'))

results = train.groupby('user_id').count()

results.row_id = 0

results.loc[count_answered_correctly_true_per_user.index, 'row_id'] = count_answered_correctly_true_per_user.row_id
(results.row_id/train.groupby('user_id').count().row_id).hist(bins=10)
(results.row_id/train.groupby('user_id').count().row_id).hist(bins=100)
((results[results.row_id<50].row_id)/(train.groupby('user_id').count()[results.row_id<50].row_id)).hist(bins=100)
(results[results.row_id>=50].row_id/train.groupby('user_id').count()[results.row_id>50].row_id).hist(bins=100)
(results[results.row_id>=50].row_id/train.groupby('user_id').count()[results.row_id>1000].row_id).hist(bins=100)
lectures = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', nrows=NROWS, dtype=dtype)

lectures = lectures[lectures['content_type_id']==1]
lectures.groupby('user_id').count().row_id.hist(bins=100)
questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', nrows=NROWS, dtype=dtype)

questions = questions.dropna()

questions = questions[questions['answered_correctly']!=-1]



lectures_count = questions.groupby('user_id').answered_correctly.mean()

lectures_count.loc[:] = 0

lectures_count.loc[lectures.groupby('user_id').count().index] = lectures.groupby('user_id').count().row_id



plt.scatter(questions.groupby('user_id').answered_correctly.mean(), lectures_count)

plt.xlabel("Correctness rate per student")

plt.ylabel("Number of lectures attended per student")

plt.show()
questions_type = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')

questions_type
print((questions_type[(questions_type.question_id != questions_type.bundle_id)]))

print('\nPercentage of questions served together: %f\n' % (len(questions_type[(questions_type.question_id != questions_type.bundle_id)])/len(questions_type)))

print('Description of unique value of bundle_id:\n%s\n' % questions_type.bundle_id.value_counts().describe())
print('Description of unique value of part:\n%s\n' % questions_type.part.value_counts().describe())
all_tags=[]

for j in [y.split() for y in questions_type['tags'].astype(str).values]:

    for i in j:

        all_tags.append(i)

print('Description of unique value of tags:\n%s\n' % pd.Series(all_tags).value_counts().describe())
lectures_type = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')

lectures_type
print('Description of unique value of part:\n%s\n' % lectures_type.part.value_counts().describe())

print('Description of unique value of tag: \n%s\n' % lectures_type.tag.value_counts().describe())

print('Value counts of the type of unique questions: \n%s' %lectures_type.type_of.value_counts())