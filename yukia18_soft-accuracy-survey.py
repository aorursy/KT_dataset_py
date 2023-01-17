from typing import List, Tuple

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline

plt.style.use("ggplot")
def soft_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[List[float], float]:

    """

    Args:

        y_true (np.ndarray): GT int indices of (N, K). N is number of samples, K is number of possible answers.

        y_pred (np.ndarray): Predicted int indices of (N). N is number of samples.

    Return:

        List of scaler values for each given GT-Prediction pairs.

        Mean of above list values.

    """

    acc = []  # type: List[float]

    for yt, yp in zip(y_true, y_pred):

        ret = 0

        for k in range(len(yt)):

            res = 0

            for j in range(len(yt)):

                if k == j:

                    continue

                res += 1 if yp == yt[j] else 0

            ret += min(1, res / 3)

        ret /= len(yt)

        acc.append(ret)

    return (acc, np.mean(acc))
y_true = np.array([

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # precise

    [0, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # normal

    [0, 0, 0, 1, 1, 1, 3, 3, 3, 3],  # vague

])



y_pred_all_clear         = np.array([0, 1, 3])

y_pred_vague_minor       = np.array([0, 1, 1])

y_pred_vague_incorrect   = np.array([0, 1, 2])

y_pred_normal_minor      = np.array([0, 0, 3])

y_pred_normal_incorrect  = np.array([0, 3, 3])

y_pred_precise_incorrect = np.array([1, 1, 3])



print("all_clear         : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_all_clear)))

print("vague_minor       : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vague_minor)))

print("vague_incorrect   : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vague_incorrect)))

print("normal_minor      : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_normal_minor)))

print("normal_incorrect  : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_normal_incorrect)))

print("precise_incorrect : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_precise_incorrect)))
# v/n = vague/normal/precise

# m/i = minor/incorrect

y_pred_vm_nm    = np.array([0, 0, 1])

y_pred_vi_nm    = np.array([0, 0, 2])

y_pred_vm_ni    = np.array([0, 3, 1])

y_pred_vi_ni    = np.array([0, 3, 2])

y_pred_vm_nm_pi = np.array([1, 0, 1])

y_pred_vi_nm_pi = np.array([1, 0, 2])

y_pred_vm_ni_pi = np.array([1, 3, 1])

y_pred_vi_ni_pi = np.array([1, 3, 2])



print("vm_nm    : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vm_nm)))

print("vi_nm    : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vi_nm)))

print("vm_ni    : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vm_ni)))

print("vi_ni    : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vi_ni)))

print("vm_nm_pi : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vm_nm_pi)))

print("vi_nm_pi : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vi_nm_pi)))

print("vm_ni_pi : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vm_ni_pi)))

print("vi_ni_pi : {}, {:.4f}".format(*soft_accuracy(y_true, y_pred_vi_ni_pi)))
K = 10

scores = []  # type: List[float]

for i in range(K + 1):

    y_true_sim = np.array([[0] * i + [1] * (K - i)])

    _, score = soft_accuracy(y_true_sim, np.array([0]))

    scores.append(score)



plt.plot(scores)

plt.xlabel("number of answers which is the same as a prediction")

plt.ylabel("soft accuracy")

plt.show()
# VQA v1.0/v2.0 testing set

# --> (Yes/No) judgement(78%) = precise

# --> (Number) counting(10%)  = (maybe) precise

# --> (Other) color(8%)/location(3%) = normal
# Model achieves 80% in Yes/No --> 答えが明らかな問題で100問中10問正解

# Model achieves 41% in Number --> 10人いたら1~2人が答えるような解答（惜しい解答)を常に出力/難しい問題全てと簡単な問題合わせて100問中40問正解