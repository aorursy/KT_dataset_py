# refrence: https://www.kaggle.com/dborkan/benchmark-kernel

class SubmetricsAUC(object):

    def __init__(self, valid_df, pred_y):

        self.identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian',

                                 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

        self.SUBGROUP_AUC = 'subgroup_auc'

        self.BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative

        self.BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive



        self.TOXICITY_COLUMN = 'target'



        self.valid_df = valid_df

        self.pred_y = pred_y

        self.model_name = 'pred'

        self.valid_df[self.model_name] = self.pred_y

        self.valid_df = self.convert_dataframe_to_bool(self.valid_df)



    def compute_auc(self):



        bias_metrics_df = self.compute_bias_metrics_for_model(self.identity_columns,

                                                              self.model_name,

                                                              self.TOXICITY_COLUMN).fillna(0)



        final_score = self.get_final_metric(bias_metrics_df, self.calculate_overall_auc())



        return final_score



    @staticmethod

    def power_mean(series, p):

        total = sum(np.power(series, p))

        return np.power(total / len(series), 1 / p)



    @staticmethod

    def calculate_auc(y_true, y_pred):

        try:

            return roc_auc_score(y_true, y_pred)

        except ValueError:

            return np.nan



    @staticmethod

    def convert_to_bool(df, col_name):

        df[col_name] = np.where(df[col_name] >= 0.5, True, False)



    def convert_dataframe_to_bool(self, df):

        bool_df = df.copy()

        for col in ['target'] + self.identity_columns:

            self.convert_to_bool(bool_df, col)

        return bool_df



    def compute_subgroup_auc(self, subgroup, label, model_name):

        subgroup_examples = self.valid_df[self.valid_df[subgroup]]

        return self.calculate_auc(subgroup_examples[label], subgroup_examples[model_name])



    def compute_bpsn_auc(self, subgroup, label, model_name):

        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""

        subgroup_negative_examples = self.valid_df[self.valid_df[subgroup] & ~self.valid_df[label]]

        non_subgroup_positive_examples = self.valid_df[~self.valid_df[subgroup] & self.valid_df[label]]

        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

        return self.calculate_auc(examples[label], examples[model_name])



    def compute_bnsp_auc(self, subgroup, label, model_name):

        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""

        subgroup_positive_examples = self.valid_df[self.valid_df[subgroup] & self.valid_df[label]]

        non_subgroup_negative_examples = self.valid_df[~self.valid_df[subgroup] & ~self.valid_df[label]]

        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)

        return self.calculate_auc(examples[label], examples[model_name])



    def compute_bias_metrics_for_model(self,

                                       subgroups,

                                       model,

                                       label_col,

                                       include_asegs=False):

        """Computes per-subgroup metrics for all subgroups and one model."""

        records = []

        for subgroup in subgroups:

            record = {'subgroup': subgroup,

                      'subgroup_size': len(self.valid_df[self.valid_df[subgroup]])

                      }



            record[self.SUBGROUP_AUC] = self.compute_subgroup_auc(subgroup, label_col, model)

            record[self.BPSN_AUC] = self.compute_bpsn_auc(subgroup, label_col, model)

            record[self.BNSP_AUC] = self.compute_bnsp_auc(subgroup, label_col, model)

            records.append(record)

        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)



    def calculate_overall_auc(self):

        true_labels = self.valid_df[self.TOXICITY_COLUMN]

        predicted_labels = self.valid_df[self.model_name]

        return roc_auc_score(true_labels, predicted_labels)



    def get_final_metric(self, bias_df, overall_auc, power=-5, weight=0.25):

        bias_score = np.average([

            self.power_mean(bias_df[self.SUBGROUP_AUC], power),

            self.power_mean(bias_df[self.BPSN_AUC], power),

            self.power_mean(bias_df[self.BNSP_AUC], power)

        ])

        return (weight * overall_auc) + ((1 - weight) * bias_score)
# # just like this

# score = SubmetricsAUC(train_df, train_preds).compute_auc()
# # seems like this 

# auc = roc_auc_score(train_y, train_preds)