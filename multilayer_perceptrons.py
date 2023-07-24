import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


class multi_perceptrons:
    def __init__(self, training_data_file_name, testing_data_file_name, epoches=50, learning_rate=0.1, momentom=0.9):
        # read raw data
        self.training_data = self._read_data(training_data_file_name)
        self.testing_data = self._read_data(testing_data_file_name)
        # number of epoches
        self.epoches = epoches
        # number of hidden neurons
        self.number_of_hidden = [20, 50, 100]
        # number of weights. Since now we have a column of target values, we should deduct one
        self.number_weights = self.training_data.shape[1] - 1
        # the number of training/testing examples
        self.training_examples = self.training_data.shape[0]
        self.testing_examples = self.testing_data.shape[0]
        # self.weights = np.random.uniform(-0.05,
        #                                  0.05, (10, self.number_weights))
        self.weights = None

    def _read_data(self, file_name):
        # read data from file and append a column bias neuron
        raw_data = pd.read_csv(file_name, header=None).values
        data_set = np.asarray(raw_data, dtype=float)
        data_set[:, 1:] = data_set[:, 1:] / 255
        data_set = np.append(data_set, np.ones((data_set.shape[0], 1)), axis=1)
        return data_set
