!pip install tensorflow-data-validation
import tensorflow_data_validation as tfdv
!pip show tensorflow_data_validation
%ls ../input
train_stats = tfdv.generate_statistics_from_csv("../input/train.csv")

tfdv.visualize_statistics(train_stats)
test_stats = tfdv.generate_statistics_from_csv("../input/test.csv")

tfdv.visualize_statistics(test_stats)