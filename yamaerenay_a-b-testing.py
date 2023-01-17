import numpy as np

from statsmodels.stats import power as sms

from scipy.stats import pearsonr, shapiro, ttest_ind

import warnings

from tqdm.notebook import tqdm

import pandas as pd

warnings.filterwarnings("ignore")



#a/b-tester

class ABTest():

    def __init__(self, alpha, power, ind_limit=0.20):

        """alpha and power required for identifying the min. sample size

        and ind_limit that defines the dependence using correlation coefficient"""

        self.alpha = alpha

        self.power = power

        self.ind_limit = ind_limit

    def fit(self, control, test, print_not_return=True):

        """min. sample size, shapiro, pearsonr and ttest, and their corresponding p-values"""

        a, b = control, test

        power, alpha = self.power, self.alpha

        p_a = a.mean()

        p_b = b.mean()

        n_a = len(a)

        n_b = len(b)

        effect_size = (p_b-p_a)/a.std()

        n_req = int(sms.TTestPower().solve_power(effect_size=effect_size, power=power, alpha=alpha))

        if len(a) > len(b):

            a = a[:len(b)]

        elif len(a) < len(b):

            b = b[:len(a)]

        stat1, p1 = shapiro(a)

        stat2, p2 = shapiro(b)

        stat3, p3 = pearsonr(a, b)

        stat4, p4 = ttest_ind(b, a)

        

        result_dict = {"power": power, "alpha": alpha, "n_req": n_req,

                       "n_control": n_a, "n_test": n_b, "shapiro_control_stat": stat1,

                       "shapiro_control_p": p1, "shapiro_test_stat": stat2, "shapiro_test_p": p2,

                       "pearsonr_stat": stat3, "pearsonr_p": p3, "ttest_stat": stat4,

                       "ttest_p": p4, "ind_limit": self.ind_limit, "very_low_number": n_req > n_a or n_req > n_b,

                       "control_is_normal": alpha < p1, "test_is_normal": alpha < p2,

                       "very_low_correlation": self.ind_limit > abs(stat3), "very_high_dependence": p3 < alpha,

                       "no_difference": p4 > alpha, "test_is_bigger": stat4 > 0, "control_is_bigger": stat4 < 0}



        accepted = True

        inter = f"""

        Power: {power:.2f}

        Alpha: {alpha:.2f}

        Required Number: {n_req:.0f}

        Control Length: {n_a}

        Test Length: {n_b}

        Control Shapiro: {stat1:.4f} ({p1:.4f})

        Test Shapiro: {stat2:.4f} ({p2:.4f})

        Correlation: {stat3:.4f} ({p3:.4f})

        TTest Stat: {stat4:.4f} ({p4:.4f})        

        

        -INTERPRETATION-

        """

        if(n_a > 5000 or n_b > 5000):

            inter += "\nWARNING: P-value may be inaccurate for n>5000"

        if(result_dict["very_low_number"]):

            inter += "\nERROR: Sample size is too LOW"

            accepted = False

        if(result_dict["control_is_normal"]):

            inter += "\nControl group is normally distributed"

        else:

            inter += "\nERROR: Control group is NOT normally distributed"

            accepted = False

        if(result_dict["test_is_normal"]):

            inter += "\nTest group is normally distributed"

        else:

            inter += "\nERROR: Test group is NOT normally distributed"

            accepted = False

        if(not(result_dict["very_high_dependence"])):

            inter += "\nControl and test group are independent"

        elif(result_dict["very_low_correlation"]):

            inter += "\nControl and test group are independent"

        else: 

            inter += f"\nERROR: Control and test group are dependent (r = {stat3})"

            accepted = False

        if(accepted):

            inter += f"\nThe conditions are great"

            if(p4 < alpha and stat4 > 0):

                inter += "\nControl and test group are different\n mean(Test) > mean(Control)"

            elif(p4 < alpha and stat4 < 0):

                inter += "\nControl and test group are different\n mean(Control) > mean(Test)"

            else:

                inter += f"\nERROR: Control and test group are NOT different (t = {stat4})"

        else:

            inter += f"\nERROR: The experiment cannot be conducted due to some previous errors"

    

        result_dict.update({"test_acceptable": accepted})

        

        if print_not_return:

            print(inter)

        else:

            return result_dict

        

#split data into n-dimensions

def nd_index(i, *args):

    vals = []

    for v in range(len(args)):

        vals.append(i%args[v])

        i = i // args[v]

    return vals



#generate fake data

def random_data(a_mean, b_mean, std, n):

    a = np.random.normal(a_mean, std, n)

    b = np.random.normal(b_mean, std, n)

    return a, b



#individual experiment results

def score_data(a_mean, diff, std, n, tester):

    b_mean = a_mean + diff 

    a, b = random_data(a_mean, b_mean, std, n)

    a_mean_actual = a.mean()

    b_mean_actual = b.mean()

    score = {"control_mean": a_mean_actual, "test_mean": b_mean_actual, "control_mean_expected": a_mean, "test_mean_expected": b_mean, "std": std}

    score.update(tester.fit(a, b, print_not_return = False))

    return score



#tries each combination of variables

def paramsgrid(a_mean, std, n, **params):

    scores = []

    size_alpha = len(params["alpha"])

    size_power = len(params["power"])

    size_diff = len(params["diff"])

    size_total = size_alpha * size_power * size_diff

    for i in tqdm(range(size_total)):

        [i_alpha, i_power, i_diff] = nd_index(i, size_alpha, size_power, size_diff)

        alpha = params["alpha"][i_alpha]

        power = params["power"][i_power]

        diff = params["diff"][i_diff]

        tester = ABTest(power=power, alpha=alpha)

        score = score_data(a_mean, diff, std, n, tester=tester)

        scores.append(score)

    return pd.DataFrame(scores)
#parameters for parameter grid

a_mean = 0.40

n = 1000

std = 0.20

power = list(np.linspace(0.80, 0.95, 16))

alpha = list(np.linspace(0.05, 0.20, 16))

diff = list(np.linspace(-0.05, 0.05, 101))



df = paramsgrid(0.10, 0.05, 1000, **{

    "alpha": alpha, "power": power, "diff": diff

})



df.to_csv("experiments(0).csv")