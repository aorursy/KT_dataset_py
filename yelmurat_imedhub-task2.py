import pandas as pd

from os.path import join

import json

from typing import List
!ls ../input/imedhub-internship/task_2/
df_doctors = pd.read_csv(join('..','input','imedhub-internship','task_2', 'doctors.csv'))

df_example = pd.read_csv(join('..','input','imedhub-internship','task_2', 'example.csv'))
df_doctors.head()
df_example.head()
df_pred = pd.DataFrame(columns=df_example.columns)
def doctors2pred(df: pd.DataFrame, img_width=1024, img_height=1024, num_doctors=3, print_rows=True) -> pd.DataFrame:

    '''

    Convert original doctors ``pd.DataFrame`` to actual value`s format

    

    Parameters

    ----------

    df: pd.DataFrame

        Original doctors predicted table

    img_width, img_height: int

        Original image dimensions

    num_doctors: int

        Total number of different doctors

    print_rows: bool

        If true, print every row in ``df``

    

    Return

    ------

    pd.DataFrame

        Reformatted ``pd.DataFrame`` table

    

    '''

    

    w_ratio, h_ratio = img_width/100, img_height/100

    preds = []

    for index, row in df.iterrows():

        name = json.loads(row.original)['name']

        for doct_ind in range(1, num_doctors+1):

            if isinstance(row[f'doctor{doct_ind}_mark'], str):

                marks = json.loads(row[f'doctor{doct_ind}_mark'])['marks']

                for mark in marks:

                    dict_pred = {'Image Index': name, 

                                 'Finding Label': mark['symptom'], 

                                 'Bbox [x': mark['x']*w_ratio, 

                                 'y': mark['y']*h_ratio,

                                 'w': mark['width']*w_ratio,

                                 'h]': mark['height']*h_ratio}

                    if print_rows:

                        print(f'img_name: {name:>5}, doct_ind: {doct_ind:>2}, mark: {mark}')

                    preds.append(dict_pred)

                    

    return pd.DataFrame(preds)
doctors_pred = doctors2pred(df_doctors)
doctors_pred.head()
def calculate_iou(x1: float, y1: float, w1: float, h1: float, x2: float, y2: float, w2: float, h2: float) -> float:

    '''

    Calculates Intersection Over Union

    

    Parameters

    ----------

    x1, y1, x2, y2 : float

        Positive real coordinates of left-bottom points of rectangles

    w1, w2: float

        Positive real widths of rectangles

    h1, h2: float

        Positive real heights of rectangles

    

    Return

    ------

    float

        Intersection Over Union [0, 1]

    

    Raises

    ------

    ValueError

        For negative parameter values

        

    '''

    

    if min(x1, y1, w1, h1, x2, y2, w2, h2) < 0:

        raise ValueError("All values should be positive")



    int_x = max(x1, x2)

    int_y = min(y1, y2)

    int_w = min(x1+w1, x2+w2) - int_x

    int_h = int_y - max(y1-h1, y2-h2)

    int_area = int_w*int_h



    if min(int_w, int_h) < 0:

        return 0

    

    actual_area = w1*h1

    pred_area = w2*h2

    

    return int_area / (actual_area + pred_area - int_area)
def calculate_iou_series(actual: pd.Series, pred: pd.Series, x='Bbox [x', y='y', w='w', h='h]') -> float:

    '''

    Returns calculated Intersection Over Union

    

    Parameters

    ----------

    actual: pd.Series

        ``pd.Series`` row of actual value

    pred: pd.Series

        ``pd.Series`` row of doctor`s predicted value

    

    x, y, w, h: str

        Labels of coordinates, widths and heights

        

    Return

    ------

    float

        Intersection Over Union

    

    '''

    return calculate_iou(actual[x], actual[y], actual[w], actual[h], pred[x], pred[y], pred[w], pred[h])
def calculate_mean_iou(actual: pd.DataFrame, pred: pd.DataFrame, ind='Image Index', label='Finding Label', x='Bbox [x', y='y', w='w', h='h]') -> List[int]:

    '''

    Returns calculated Intersection Over Union

    

    Parameters

    ----------

    actual: pd.DataFrame

        Table of actual value

    pred: pd.DataFrame

        Row of doctor`s predicted value

    

    x, y, w, h: str

        Labels of coordinates, widths and heights

        

    Return

    ------

    List[int]

        List of Mean IoU on doctors predicted values for every actual value

        

        -1. : No predicted values for actual

    

    '''



    means = []

    for _, actual_row in actual.iterrows():

        ind_bool = pred[ind]==actual_row[ind]

        label_bool = pred[label].str.lower()==actual_row[label].lower()

        pred_rows = pred[ind_bool & label_bool]



        if not pred_rows.shape[0]:

            mean = -1.

        else:

            mean = sum([calculate_iou_series(actual_row, pred_row, x, y, w, h) for _, pred_row in pred_rows.iterrows()]) / pred_rows.shape[0]

            

        means.append(mean)

    return means
calculate_mean_iou(df_example, doctors_pred)