from pathlib import Path

from typing import List, Tuple, Dict, Union

from functools import partial



import time

import os

import sys

import random

import shutil



from sklearn.model_selection import KFold



import numpy as np 

import pandas as pd
import chainer

from chainer import links, functions

from chainer import datasets, iterators, optimizers, serializers

from chainer import training, reporter, cuda

from chainercv.links import PickableSequentialChain
class LinearActiv(chainer.Chain):

    """linear -> activation (-> batch norm -> dropout)"""



    def __init__(

        self, in_size: int, out_size: int,

        dropout_rate=None, use_bn=False, activ=functions.relu

    ) -> None:

        """Initialize."""

        super(LinearActiv, self).__init__()

        layers = chainer.Sequential(links.Linear(in_size, out_size))

        if activ is not None:

            layers.append(activ)

        if use_bn:

            layers.append(links.BatchNormalization(out_size))

        if dropout_rate is not None:

            layers.append(partial(functions.dropout, ratio=dropout_rate))



        with self.init_scope():

            self.la = layers



    def __call__(self, x: chainer.Variable) -> chainer.Variable:

        """Forward."""

        return self.la(x)

    



class MLP(chainer.Chain):

    """Multi Layer Perceptron."""



    def __init__(

        self, in_dim: int, hidden_dims: List[int],

        drop_rates: List[float]=None, use_bn=False, use_tail_as_out=True

    ) -> None:

        """initialize."""

        super(MLP, self).__init__()

        hidden_dims = [in_dim] + hidden_dims

        drop_rates = [None] * len(hidden_dims) if drop_rates is None else drop_rates

        layers = [

            LinearActiv(

                hidden_dims[i], hidden_dims[i + 1], drop_rates[i], use_bn)

            for i in range(len(hidden_dims) - 2)]



        if use_tail_as_out:

            layers.append(links.Linear(hidden_dims[-2], hidden_dims[-1]))

        else:

            layers.append(

                LinearActiv(

                    hidden_dims[-2], hidden_dims[-1], drop_rates[-1], use_bn))



        with self.init_scope():

            self.layers = chainer.Sequential(*layers)



    def __call__(self, x: chainer.Variable) -> chainer.Variable:

        """Forward."""

        return self.layers(x)





class CustomMLP(chainer.Chain):

    """Simple MLP model."""

    

    def __init__(

        self, left_mlp: MLP, right_mlp: MLP, tail_mlp: MLP

    ) -> None:

        """Initialize."""

        super(CustomMLP, self).__init__()

        with self.init_scope():

            self.left = left_mlp

            self.right = right_mlp

            self.tail = tail_mlp

        

    def __call__(self, x_left: chainer.Variable, x_right: chainer.Variable) -> chainer.Variable:

        """Forward."""

        h_left = self.left(x_left)

        h_right = self.right(x_right)

        h = functions.concat([h_left, h_right])

        h = self.tail(h)

        return h
class Regressor(links.Classifier):

    """Wrapper for regression model."""



    def __init__(

        self, predictor, lossfun, evalfun_dict

    ):

        """Initialize"""

        super(Regressor, self).__init__(predictor, lossfun)

        self.compute_accuracy = False

        self.evalfun_dict = evalfun_dict

        for name, func in self.evalfun_dict.items():

            setattr(self, name, None)

            

    def evaluate(self, *in_arrs: Tuple[chainer.Variable]) -> None:

        """Calc loss and evaluation metric."""

        for name in self.evalfun_dict.keys():

            setattr(self, name, None)

        loss = self(*in_arrs)



        for name, evalfun in self.evalfun_dict.items():

            setattr(self, name, evalfun(self.y, in_arrs[-1]))

            reporter.report({name: getattr(self, name)}, self)

        del loss
def normalized_absolute_error(y_pred: chainer.Variable, t: np.ndarray):

    """

    \sum_{i} |y_pred_{i} - t_{i}| / \sum_{i} t_{i}

    """

    return functions.sum(functions.absolute(y_pred - t)) / functions.sum(t)





class WeightedNormalizedAbsoluteError:

    """Metric for this competition"""

    

    def __init__(self, weights: List[float]=[.3, .175, .175, .175, .175]):

        """Initialize."""

        self.weights = weights

        self.pred_num = len(weights)

        

    def __call__(self, y_pred: chainer.Variable, t: np.ndarray) ->  chainer.Variable:

        """Forward."""

        loss = 0

        for i, weight in enumerate(self.weights):

            loss += weight * normalized_absolute_error(y_pred[:, i], t[:, i])

            

        return loss

    



class SelectNormalizedAbsoluteError:

    """For checking each features loss"""

    

    def __init__(self, index: int):

        """Initialize."""

        self.index = index

        

    def __call__(self, y_pred: chainer.Variable, t: np.ndarray) ->  chainer.Variable:

        """Forward."""

        return normalized_absolute_error(y_pred[:, self.index], t[:, self.index])
def set_random_seed(seed=42):

    """Fix Seeds."""

    # set Python random seed

    random.seed(seed)



    # set NumPy random seed

    np.random.seed(seed)



    # set Chainer(CuPy) random seed

    cuda.cupy.random.seed(seed)
COMPETITION_NAME = "trends-assessment-prediction"

ROOT = Path(".").resolve().parents[0]



INPUT_ROOT = ROOT / "input"

RAW_DATA = INPUT_ROOT / COMPETITION_NAME

TRAIN_IMAGES = RAW_DATA / "fMRI_train"

TEST_IMAGES = RAW_DATA / "fMRI_test"
fnc = pd.read_csv(RAW_DATA / "fnc.csv")

icn_numbers = pd.read_csv(RAW_DATA / "ICN_numbers.csv")

loading = pd.read_csv(RAW_DATA / "loading.csv")

reveal_ID_site2 = pd.read_csv(RAW_DATA / "reveal_ID_site2.csv")



train_scores = pd.read_csv(RAW_DATA / "train_scores.csv")

sample_sub = pd.read_csv(RAW_DATA / "sample_submission.csv")
sample_sub.shape[0] / 5 
# # init model

def init_model(is_train=True):

    model = CustomMLP(

        left_mlp=MLP(

            in_dim=1378, hidden_dims=[1024, 768], drop_rates=[0.5, 0.5], use_tail_as_out=False),

        right_mlp=MLP(

            in_dim=26, hidden_dims=[64, 768], drop_rates=[0.5, 0.5], use_tail_as_out=False),

        tail_mlp=MLP(

            in_dim=1536, hidden_dims=[1024, 5], drop_rates=[0.5, 0.0]),

    )

    if not is_train:

        return model



    # # set trainning wrapper

    train_model = Regressor(

        predictor=model,

        lossfun=WeightedNormalizedAbsoluteError(weights=[.3, .175, .175, .175, .175]),

        evalfun_dict={

            "NAE_Age": SelectNormalizedAbsoluteError(0),

            "NAE_Domain1Var1": SelectNormalizedAbsoluteError(1),

            "NAE_Domain1Var2": SelectNormalizedAbsoluteError(2),

            "NAE_Domain2Var1": SelectNormalizedAbsoluteError(3),

            "NAE_Domain2Var2": SelectNormalizedAbsoluteError(4)}

    )

    return train_model
DEVICE = 0
def create_trainer(train_model, train_dataset, val_dataset, output_dir, device):

    # # set optimizer

    optimizer = optimizers.AdamW(alpha=0.001, weight_decay_rate=0.0)

    optimizer.setup(train_model)

    

    # # make iterator

    train_iter = iterators.MultiprocessIterator(

        train_dataset, 64, n_processes=2)

    val_iter = iterators.MultiprocessIterator(

        val_dataset, 64, repeat=False, shuffle=False, n_processes=2)

    

    # # init trainer

    updater = training.StandardUpdater(train_iter, optimizer, device=device)



    stop_trigger = training.triggers.EarlyStoppingTrigger(

        check_trigger=(1, 'epoch'), monitor='val/main/loss', mode="min",

        patients=20, max_trigger=(200, 'epoch'), verbose=True)



    trainer = training.trainer.Trainer(

        updater, stop_trigger=stop_trigger, out=output_dir)

    

    # # set extentions

    lr_attr_name = "alpha"

    log_trigger = (1, "epoch")

    logging_attributes = [

        "epoch", "elapsed_time", "main/loss", "val/main/loss",

        "val/main/NAE_Age",

        "val/main/NAE_Domain1Var1", "val/main/NAE_Domain1Var2",

        "val/main/NAE_Domain2Var1", "val/main/NAE_Domain2Var2"]



    # # # evaluator

    eval_target = trainer.updater.get_optimizer('main').target

    trainer.extend(

        training.extensions.Evaluator(

            val_iter, eval_target, device=device, eval_func=eval_target.evaluate),

        name='val',trigger=(1, 'epoch'))



    # # # log.

    trainer.extend(

        training.extensions.observe_lr(observation_key=lr_attr_name), trigger=log_trigger)

    trainer.extend(

        training.extensions.LogReport(logging_attributes, trigger=log_trigger), trigger=log_trigger)

    trainer.extend(training.extensions.PrintReport(logging_attributes), trigger=log_trigger)



    # # # save snapshot

    trainer.extend(

        training.extensions.snapshot_object(

            trainer.updater.get_optimizer('main').target.predictor,

            'model_snapshot_{.updater.epoch}.npz'),

        trigger=training.triggers.MinValueTrigger("val/main/loss", (1, "epoch")))

    

    return trainer
val_score_list = []
features = pd.merge(fnc,loading, on="Id", how="inner") 

train_all = train_scores.merge(features, on="Id", how="left")



# #For convenience in trainning, fill NA by mean values. 



for i in range(5):

    train_all.iloc[:, i + 1] = train_all.iloc[:, i + 1].fillna(train_all.iloc[:, i + 1].mean())



kf = KFold(n_splits=5, shuffle=True, random_state=1086)

train_val_splits = list(kf.split(X=train_scores.Id))
fold_id = 0

train_index, val_index = train_val_splits[fold_id]

train = train_all.iloc[train_index]

val = train_all.iloc[val_index]



train_dataset = datasets.TupleDataset(

    train.iloc[:, 6:1384].values.astype("f"),  # fnc

    train.iloc[:, 1384:].values.astype("f"),  # loading

    train.iloc[:, 1:6].values.astype("f"),  # label

)

val_dataset = datasets.TupleDataset(

    val.iloc[:, 6:1384].values.astype("f"),  # fnc

    val.iloc[:, 1384:].values.astype("f"),  # loading

    val.iloc[:, 1:6].values.astype("f"),  # label

)

set_random_seed(1086)

train_model = init_model()

trainer = create_trainer(train_model, train_dataset, val_dataset, "training_result_fold{}".format(fold_id), DEVICE)

trainer.run()
best_epoch = trainer.updater.epoch - trainer.stop_trigger.count

print(best_epoch, trainer.stop_trigger.best)

val_score_list.append([fold_id, trainer.stop_trigger.best, best_epoch,])

shutil.copyfile(

    "training_result_fold{}/model_snapshot_{}.npz".format(fold_id, best_epoch), "best_model_fold{},npz".format(fold_id))
fold_id = 1

train_index, val_index = train_val_splits[fold_id]

train = train_all.iloc[train_index]

val = train_all.iloc[val_index]



train_dataset = datasets.TupleDataset(

    train.iloc[:, 6:1384].values.astype("f"),  # fnc

    train.iloc[:, 1384:].values.astype("f"),  # loading

    train.iloc[:, 1:6].values.astype("f"),  # label

)

val_dataset = datasets.TupleDataset(

    val.iloc[:, 6:1384].values.astype("f"),  # fnc

    val.iloc[:, 1384:].values.astype("f"),  # loading

    val.iloc[:, 1:6].values.astype("f"),  # label

)

set_random_seed(1086)

train_model = init_model()

trainer = create_trainer(train_model, train_dataset, val_dataset, "training_result_fold{}".format(fold_id), DEVICE)

trainer.run()
best_epoch = trainer.updater.epoch - trainer.stop_trigger.count

print(best_epoch, trainer.stop_trigger.best)

val_score_list.append([fold_id, trainer.stop_trigger.best, best_epoch,])

shutil.copyfile(

    "training_result_fold{}/model_snapshot_{}.npz".format(fold_id, best_epoch), "best_model_fold{},npz".format(fold_id))
fold_id = 2

train_index, val_index = train_val_splits[fold_id]

train = train_all.iloc[train_index]

val = train_all.iloc[val_index]



train_dataset = datasets.TupleDataset(

    train.iloc[:, 6:1384].values.astype("f"),  # fnc

    train.iloc[:, 1384:].values.astype("f"),  # loading

    train.iloc[:, 1:6].values.astype("f"),  # label

)

val_dataset = datasets.TupleDataset(

    val.iloc[:, 6:1384].values.astype("f"),  # fnc

    val.iloc[:, 1384:].values.astype("f"),  # loading

    val.iloc[:, 1:6].values.astype("f"),  # label

)

set_random_seed(1086)

train_model = init_model()

trainer = create_trainer(train_model, train_dataset, val_dataset, "training_result_fold{}".format(fold_id), DEVICE)

trainer.run()
best_epoch = trainer.updater.epoch - trainer.stop_trigger.count

print(best_epoch, trainer.stop_trigger.best)

val_score_list.append([fold_id, trainer.stop_trigger.best, best_epoch,])

shutil.copyfile(

    "training_result_fold{}/model_snapshot_{}.npz".format(fold_id, best_epoch), "best_model_fold{},npz".format(fold_id))
fold_id = 3

train_index, val_index = train_val_splits[fold_id]

train = train_all.iloc[train_index]

val = train_all.iloc[val_index]



train_dataset = datasets.TupleDataset(

    train.iloc[:, 6:1384].values.astype("f"),  # fnc

    train.iloc[:, 1384:].values.astype("f"),  # loading

    train.iloc[:, 1:6].values.astype("f"),  # label

)

val_dataset = datasets.TupleDataset(

    val.iloc[:, 6:1384].values.astype("f"),  # fnc

    val.iloc[:, 1384:].values.astype("f"),  # loading

    val.iloc[:, 1:6].values.astype("f"),  # label

)

set_random_seed(1086)

train_model = init_model()

trainer = create_trainer(train_model, train_dataset, val_dataset, "training_result_fold{}".format(fold_id), DEVICE)

trainer.run()
best_epoch = trainer.updater.epoch - trainer.stop_trigger.count

print(best_epoch, trainer.stop_trigger.best)

val_score_list.append([fold_id, trainer.stop_trigger.best, best_epoch,])

shutil.copyfile(

    "training_result_fold{}/model_snapshot_{}.npz".format(fold_id, best_epoch), "best_model_fold{},npz".format(fold_id))
fold_id = 4

train_index, val_index = train_val_splits[fold_id]

train = train_all.iloc[train_index]

val = train_all.iloc[val_index]



train_dataset = datasets.TupleDataset(

    train.iloc[:, 6:1384].values.astype("f"),  # fnc

    train.iloc[:, 1384:].values.astype("f"),  # loading

    train.iloc[:, 1:6].values.astype("f"),  # label

)

val_dataset = datasets.TupleDataset(

    val.iloc[:, 6:1384].values.astype("f"),  # fnc

    val.iloc[:, 1384:].values.astype("f"),  # loading

    val.iloc[:, 1:6].values.astype("f"),  # label

)

set_random_seed(1086)

train_model = init_model()

trainer = create_trainer(train_model, train_dataset, val_dataset, "training_result_fold{}".format(fold_id), DEVICE)

trainer.run()
best_epoch = trainer.updater.epoch - trainer.stop_trigger.count

print(best_epoch, trainer.stop_trigger.best)

val_score_list.append([fold_id, trainer.stop_trigger.best, best_epoch,])

shutil.copyfile(

    "training_result_fold{}/model_snapshot_{}.npz".format(fold_id, best_epoch), "best_model_fold{},npz".format(fold_id))
pd.DataFrame(

    val_score_list,

    columns=["fold", "score", "best_epoch"])
def inference_test_data(

    model: Union[chainer.Chain, PickableSequentialChain],

    test_iter: chainer.iterators.MultiprocessIterator, gpu_device: int=-1

) -> Tuple[np.ndarray]:

    """Oridinary Inference."""

    test_pred_list = []

    test_label_list = []

    iter_num = 0

    epoch_test_start = time.time()



    while True:

        test_batch = test_iter.next()

        iter_num += 1

        print("\rtmp_iteration: {:0>5}".format(iter_num), end="")

        in_arrays = chainer.dataset.concat_examples(test_batch, gpu_device)



        # Forward the test data

        with chainer.no_backprop_mode() and chainer.using_config("train", False):

            prediction_test = model(*in_arrays[:-1])

            test_pred_list.append(prediction_test)

            test_label_list.append(in_arrays[-1])

            prediction_test.unchain_backward()



        if test_iter.is_new_epoch:

            print(" => test end: {:.2f} sec".format(time.time() - epoch_test_start))

            test_iter.reset()

            break



    test_pred_all = cuda.to_cpu(functions.concat(test_pred_list, axis=0).data)

    test_label_all = cuda.to_cpu(functions.concat(test_label_list, axis=0).data)

    del test_pred_list

    del test_label_list

    return test_pred_all, test_label_all
sample_sub.head()
test = pd.DataFrame({}, columns=train_scores.columns.tolist())

test["Id"] = sample_sub["Id"].apply(lambda x: int(x.split('_')[0])).unique()

test = test.fillna(-1)

test = test.merge(features, on="Id", how="left")



test_label = test.iloc[:, 1:6].values.astype("f")

test_fnc = test.iloc[:, 6:1384].values.astype("f")

test_loading = test.iloc[:, 1384:].values.astype("f")



test_dataset = datasets.TupleDataset(test_fnc, test_loading, test_label)
test_preds_arr = np.zeros((5, len(test), 5))

for fold_id in range(5):

    test_iter = iterators.MultiprocessIterator(

        test_dataset, 64, repeat=False, shuffle=False, n_processes=2)

    model = init_model(is_train=False)

    model.to_gpu(DEVICE)

    serializers.load_npz("best_model_fold{},npz".format(fold_id), model)

    test_pred, _ = inference_test_data(model, test_iter, DEVICE)

    test_preds_arr[fold_id] =  test_pred

    del test_iter

    del test_pred

    del model
test_pred = test_preds_arr.mean(axis=0)
test_pred.shape
5877 * 5
test_sub = test.iloc[:, :6].copy()

test_sub.iloc[:, 1:] = test_pred



test_sub = pd.melt(test_sub, id_vars="Id", value_name="Predicted")

test_sub["Id"] = test_sub["Id"].astype("str") + "_" +  test_sub["variable"]



test_sub = pd.merge(sample_sub["Id"], test_sub[["Id", "Predicted"]], how="left")

test_sub.to_csv('submission.csv', index=False)
test_sub.Predicted.isnull().value_counts()
test_sub.head()
test_sub.tail()