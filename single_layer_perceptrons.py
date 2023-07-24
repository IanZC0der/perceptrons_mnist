import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


class perceptron:
    def __init__(self, training_data_file_name, testing_data_file_name, epoches=70):
        self.learning_rates = [0.001, 0.01, 0.1]
        self.training_data = self._read_data(training_data_file_name)
        self.testing_data = self._read_data(testing_data_file_name)
        self.epoches = 70
        self.number_weights = self.training_data.shape[1] - 1
        self.training_examples = self.training_data.shape[0]
        self.testing_examples = self.testing_data.shape[0]
        self.weights = np.random.uniform(-0.05,
                                         0.05, (10, self.number_weights))
        self.results = {key: [[], []] for key in self.learning_rates}
        self.trained_weights = {key: [] for key in self.learning_rates}

    def _read_data(self, file_name):
        raw_data = pd.read_csv(file_name, header=None).values
        data_set = np.asarray(raw_data, dtype=float)
        data_set[:, 1:] = data_set[:, 1:] / 255
        data_set = np.append(data_set, np.ones((data_set.shape[0], 1)), axis=1)
        return data_set

    def train(self):
        for learning_rate in self.learning_rates:
            self.weights = np.random.uniform(-0.05,
                                             0.05, (10, self.number_weights))
            for epoch in range(self.epoches):
                for i in range(self.testing_examples):
                    targets = np.zeros(10, int)
                    targets[int(self.training_data[i, 0])] = 1
                    outputs = np.dot(self.weights, self.training_data[i, 1:])
                    outputs[outputs > 0] = 1.0
                    outputs[outputs <= 0] = 0.0

                    self.weights += learning_rate * \
                        np.outer(np.subtract(targets, outputs),
                                 self.training_data[i, 1:])
                self.results[learning_rate][0].append(
                    self._test(self.training_examples, self.training_data))
                self.results[learning_rate][1].append(
                    self._test(self.testing_examples, self.testing_data))
            # save the weigths after 70 epoches
            self.trained_weights[learning_rate] = self.weights

    def _test(self, number_examples, data):
        outputs = np.dot(data[:, 1:], self.weights.T)
        correct_number = 0
        for i in range(number_examples):
            index = np.argmax(outputs[i, :])
            if index == data[i, 0]:
                correct_number += 1
        return correct_number/number_examples

    def confusion_matrix(self):
        for learning_rate in self.learning_rates:
            initial_matrix = np.zeros((10, 10), dtype=int)
            outputs = np.dot(
                self.testing_data[:, 1:], self.trained_weights[learning_rate].T)
            for i in range(self.testing_examples):
                index = int(np.argmax(outputs[i, :]))
                initial_matrix[int(self.testing_data[i, 0]), index] += 1

            fig, ax = plot_confusion_matrix(
                conf_mat=initial_matrix, colorbar=True, fontcolor_threshold=1, cmap="summer")
            plt.title(f"Confusion Matrix at Learning Rate {learning_rate}")
            plt.show()

    def plotting(self):
        fig, axs = plt.subplots(3, sharex=True, sharey=True)
        x_axis = np.arange(1, self.epoches + 1, 1)
        for i in range(len(self.learning_rates)):
            axs[i].plot(x_axis, self.results[self.learning_rates[i]]
                        [0], color='r', label='train')
            axs[i].plot(x_axis, self.results[self.learning_rates[i]]
                        [1], color='g', label='test')

        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()


def main():
    training_name = "mnist_train.csv"
    testing_name = "mnist_test.csv"
    model = perceptron(training_name, testing_name)
    model.train()
    model.plotting()
    model.confusion_matrix()


if __name__ == "__main__":
    main()
