import numpy as np

import os

import pandas as pd

import json



if os.path.exists('/kaggle/input'):

    import sys

    sys.path.append('/kaggle/input/arclib4/')



from arclib.dsl import *

from arclib.util import evaluate_predict_func, test_path, get_string, data_path

from arclib.dsl import Task, unique_arrays

from arclib.check import  check_output_color_from_input





def check_output_in_candidates(output, candidates):

    output_is_candidate = False



    for candidate in candidates:

        if output.shape == candidate.shape:

            if (output == candidate).all():

                output_is_candidate = True

    return output_is_candidate





def predict_part(task, get_candidates, train_object_maps=None, train_bg_colors=None):



    part_task = True

    for i, (input, output) in enumerate(task.pairs):

        input = np.array(input)

        output = np.array(output)



        candidates = get_candidates(input, object_maps=train_object_maps[i], bg_color=train_bg_colors[i])



        if candidates:

            if check_output_in_candidates(output, candidates) == False:

                part_task = False

                break

        else:

            part_task = False

            break



    all_input_predictions = []

    if part_task:

        for input in task.test_inputs:

            test_candidates = get_candidates(input)

            predictions = test_candidates #[:3]

            predictions = unique_arrays(predictions)

            predictions = sorted(predictions, key= lambda x: x.shape[0] * x.shape[1], reverse=True)

            all_input_predictions.append(predictions)



    else:

        all_input_predictions = []

    return all_input_predictions





def get_cropped_object(array, object_map, bg_color=None):

    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)

    return array[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]





def keep_one_object(array, object_map, bg_color=None):

    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)

    if bg_color is None:

        bg_color = detect_bg_(array)

    output_ = np.full_like(array, bg_color)

    output_[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1] = array[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]

    return output_





def get_cropped_objects(array, get_object_maps=None, object_maps=None, augment=None, bg_color=None):

    if object_maps is None:

        object_maps = get_object_maps(array)

    objects = [augment(get_cropped_object(array, object_map)) for object_map in object_maps if np.count_nonzero(object_map) > 0]

    return objects





def get_inputs_with_one_object(array, get_object_maps=None, object_maps=None, augment=None, bg_color=None):

    if object_maps is None:

        object_maps = get_object_maps(array)

    objects = [augment(keep_one_object(array, object_map, bg_color=bg_color)) for object_map in object_maps if np.count_nonzero(object_map) > 0]

    return objects





get_object_map_funcs = [ get_objects_by_connectivity_, partial(get_objects_by_connectivity_, touch='corner'),

                    get_objects_by_color_and_connectivity_, partial(get_objects_by_color_and_connectivity_, touch='corner'), get_objects_by_color_,

                    get_objects_rectangles, partial(get_objects_rectangles, direction='horisontal'), get_objects_rectangles_without_noise, get_objects_rectangles_without_noise_without_padding]



def predict_part_types(task):



    predictions = []



    if check_output_color_from_input(task):



        bg_colors = [detect_bg_(input_) for input_ in task.inputs]

        for i, get_object_maps in enumerate(get_object_map_funcs):



            object_maps_list = [get_object_maps(input_, bg_color=bg_color) for input_, bg_color in zip(task.inputs, bg_colors)]

            for get_object_func in (get_cropped_objects, get_inputs_with_one_object):

                for augment in simple_output_process_options:

                    get_candidates = partial(get_object_func, get_object_maps=get_object_maps, augment=augment)

                    predictions = predict_part(task, get_candidates=get_candidates, train_object_maps=object_maps_list, train_bg_colors=bg_colors)

                    if predictions:

                        break

                if predictions:

                    break

            if predictions:

                break





    return predictions





def submit(predict):

    submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')

    submission['output'] = ''

    test_fns = sorted(os.listdir(test_path))

    count = 0

    for fn in test_fns:

        fp = test_path / fn

        with open(fp, 'r') as f:

            task = Task(json.load(f))

            all_input_preds = predict(task)

            if all_input_preds:

                print(fn)

                count += 1



            for i, preds in enumerate(all_input_preds):

                output_id = str(fn.split('.')[-2]) + '_' + str(i)

                string_preds = [get_string(pred) for pred in preds[:3]]

                pred = ' '.join(string_preds)

                submission.loc[output_id, 'output'] = pred



    print(count)

    submission.to_csv('submission.csv')





def main():

    #evaluate_predict_func(predict_part_types)

    submit(predict_part_types)





if __name__ == '__main__':

    main()