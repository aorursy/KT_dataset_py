# # First prediction

# # The results are saved in the file submission_old.csv, fields: id, label_old, pred_old



# submission_old = pd.DataFrame.from_dict({'id': test_img_paths, 'label_old': test_predictions, 'pred_old': test_predictions})



# submission_old['label_old'] = submission_old['label_old'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')

# submission_old['id'] = submission_old['id'].str.replace('/kaggle/working/test/unknown/', '')

# submission_old['id'] = submission_old['id'].str.replace('.jpg', '')

# submission_old.set_index('id', inplace=True)



# submission_old.to_csv('submission_old.csv')

# submission_old.head()
# # Second prediction

# # For example, we change the learning speed from 0.0001 to 0.00015. Run all the code in the first block again. 

# # We skip the block with the first prediction (we do not start it), otherwise the previous result in the submission_old.csv file will be erased.

# # The new results will be saved in the file submission_new.csv, fields: id, label_new, pred_new



# submission_new = pd.DataFrame.from_dict({'id': test_img_paths, 'label_new': test_predictions, 'pred_new': test_predictions})



# submission_new['label_new'] = submission_new['label_new'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')

# submission_new['id'] = submission_new['id'].str.replace('/kaggle/working/test/unknown/', '')

# submission_new['id'] = submission_new['id'].str.replace('.jpg', '')

# submission_new.set_index('id', inplace=True)



# submission_new.to_csv('submission_new.csv')

# submission_new.head()
import pandas as pd



df_old = pd.read_csv("../input/submission/submission_new.csv")

df_new = pd.read_csv("../input/submission/submission_old.csv")
df_test = df_old.merge(df_new, on='id')

subset = df_test.query("label_old != label_new")

subset.head(100)
import os

print(os.listdir("../input"))
import zipfile

with zipfile.ZipFile('../input/platesv2/plates.zip', 'r') as zip_obj:

    zip_obj.extractall('/kaggle/working/')



print(os.listdir('/kaggle/working/'))

print(os.listdir('/kaggle/working/plates/'))
from PIL import Image

import matplotlib.pyplot as plt



n = len(subset)+1

s = 0

plt.figure(figsize=(20, round(n*3)))

for i, l in zip(subset['id'], subset['label_new']):

    i = str(i).zfill(4)

    img = Image.open(f'/kaggle/working/plates/test/{i}.jpg')

    plt.subplot(round(n/2), 3, s + 1)

    s += 1

    plt.imshow(img)

    plt.title(f'{i} {l}')

    plt.xticks([])

    plt.yticks([])

plt.show()
midlle_pred = 0.5

step = 0.05



dirty_old = df_test.query("label_old == 'dirty' & pred_old < @midlle_pred + @step")

cleaned_old = df_test.query("label_old == 'cleaned' & pred_old > @midlle_pred - @step")



dirty_new = df_test.query("label_new == 'dirty' & pred_new < @midlle_pred + @step")

cleaned_new = df_test.query("label_new == 'cleaned' & pred_new > @midlle_pred - @step")



fig, ax = plt.subplots()



# colors: 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'



ax.scatter(dirty_old.pred_old, dirty_old.id, c = 'm', alpha = 0.5)

ax.scatter(cleaned_old.pred_old, cleaned_old.id, c = 'y', alpha = 0.5)



ax.scatter(dirty_new.pred_new, dirty_new.id, c = 'r')

ax.scatter(cleaned_new.pred_new, cleaned_new.id, c = 'g')



ax.plot([midlle_pred, midlle_pred], [0, 744], c = 'b')

#ax.plot([0.795, 0.795], [0, 744], c = 'b')



ax.minorticks_on()

ax.grid(which='major', color = 'k', linewidth = 0.5)

ax.grid(which='minor', color = 'k', linestyle = ':')



fig.set_figwidth(16)

fig.set_figheight(12)



# for x, y in zip(dirty_old.pred, dirty_old.id):

#     plt.text(x, y, y)

# for x, y in zip(cleaned_old.pred, cleaned_old.id):

#     plt.text(x, y, y)

    

for x, y in zip(dirty_new.pred_new, dirty_new.id):

    plt.text(x, y, y)

for x, y in zip(cleaned_new.pred_new, cleaned_new.id):

    plt.text(x, y, y)



plt.show()
!rm -rf train val test plates __MACOSX