import json
import pandas as pd
import path

with open("./instances_train2014.json") as i:
    d = json.load(i)
d.keys()
categories = pd.DataFrame(d['categories'])
annotations = pd.DataFrame(d['annotations'])
images = pd.DataFrame(d['images'])
images.head()
categories.head()
annotations.head()

data = (annotations
        .merge(categories, how='left', left_on='category_id', right_on='id')
        .merge(images, how='left', left_on='image_id', right_on='id'))
data.head()
largest_bbox = data.pivot_table(index='file_name', values='area', aggfunc=max).reset_index()
largest_bbox = largest_bbox.merge(data[['area', 'bbox', 'image_id', 'file_name', 'name']], how='left')
largest_bbox.head()
def bb_hw_pandas(x):
    return [x[1], x[0], x[1]+x[3]-1, x[0]+x[2]-1]

largest_bbox['bbox_new'] = largest_bbox['bbox'].apply(lambda x: bb_hw_pandas(x))
largest_bbox['bbox_str'] = largest_bbox['bbox_new'].apply(lambda x: ' '.join(str(y) for y in x))
largest_bbox.head()

f = "bbox_dataset.csv"
largest_bbox[['file_name', 'bbox_str']].to_csv(f, index=False)
largest_bbox[['file_name', 'bbox_str']].head()
