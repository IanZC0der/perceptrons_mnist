import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


class multi_perceptrons:
    def __init__(self, training_data_file_name, testing_data_file_name, epoches=50, learning_rate=0.1, momentom=0.9):
        # read raw data
        self.training_data = self._read_data(training_data_file_name)
        self.testing_data = self._read_data(testing_data_file_name)
        self.learning_rate = 0.1
        # number of epoches
        self.epoches = epoches
        # number of hidden neurons
        self.number_of_hidden_neurons = [20, 50, 100]
        # number of weights. Since now we have a column of target values, we should deduct one
        self.number_weights_i = self.training_data.shape[1] - 1
        # number of hidden of hidden neurons
        self.number_weights_h = None
        # the number of training/testing examples
        self.training_examples = self.training_data.shape[0]
        self.testing_examples = self.testing_data.shape[0]
        # self.weights = np.random.uniform(-0.05,
        #                                  0.05, (10, self.number_weights))
        self.weights_ih = None
        self.prev_change_weights_ih = None
        self.weights_ho = None
        self.prev_change_weights_ho = None
        self.momentom = momentom

    def _read_data(self, file_name):
        # read data from file and append a column bias neuron
        raw_data = pd.read_csv(file_name, header=None).values
        data_set = np.asarray(raw_data, dtype=float)
        data_set[:, 1:] = data_set[:, 1:] / 255
        data_set = np.append(data_set, np.ones((data_set.shape[0], 1)), axis=1)
        return data_set

    def _sigmoid(self, data_set):
        return 1 / (1 + np.exp(data_set))

    def train(self):
        for number_of_hidden in self.number_of_hidden_neurons:
            # we should add a bias weight
            self.number_weights_h = number_of_hidden + 1
            self.weights_ih = np.random.uniform(-0.05, 0.05,
                                                (number_of_hidden, self.number_weights))
            self.weights_ho = np.random.uniform(-0.05,
                                                0.05, (10, self.number_weights_h))
            self.prev_change_weights_ho = np.zeros((10, self.number_weights_h))
            self.prev_change_weights_ih = np.zeros(
                (number_of_hidden, self.number_weights))
            for epoch in range(self.epoches):
                for i in range(self.training_examples):
                    # create an array of 0.1. The target output value for the perceptrons that should fire should be 0.9. 0.1 otherwise
                    targets = np.ones(10, float)/10
                    targets[int(self.training_data(i, 0))] = 0.9
                    # calculate the outputs of the hidden neurons
                    hidden_outputs = self._sigmoid(
                        np.dot(self.weights_ih, self.training_data[i, 1:]))
                    # append the bias value
                    hidden_outputs_append_ones = np.append(
                        hidden_outputs, [1])
                    outputs = self._sigmoid(
                        np.dot(self.weights_ho, hidden_outputs))
                    # calculate errors
                    output_errors = outputs * \
                        (1 - outputs) * (targets - outputs)
                    hidden_errors = hidden_outputs * \
                        (1 - hidden_outputs) * \
                        np.dot(output_errors, self.weights_ho[:, :-1])
                    # calculate weight changes
                    current_weight_changes_ho = self.learning_rate * \
                        np.outer(output_errors, hidden_outputs_append_ones) + \
                        self.momentom * self.prev_change_weights_ho
                    self.prev_change_weights_ho = current_weight_changes_ho
                    current_weight_changes_ih = self.learning_rate * \
                        np.outer(hidden_errors, self.training_data[i, :])
