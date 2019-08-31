import time

from sklearn.model_selection import KFold
import numpy as np
import copy
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection

def step_derivative(y):
    return 0


def sigmoid_derivative(z):
    return z * (1 - z)


def tanh_derivative(x):
    return 1 - x ** 2


class Perceptron:
    def __init__(self, list_w, bias, learning_rate=0.1, epochs=100, activation_function='sigmoid'):
        self.list_w = list_w
        self.bias = bias
        self.activation_function = activation_function
        self.learning_rate = learning_rate  # eta
        self.epochs = epochs
        self.output = 0
        self.delta = 0

    def feed(self, inp):
        x = np.array(inp)
        w = np.array(self.list_w)
        y = np.sum(w * x)
        if self.activation_function == 'sigmoid':
            self.output = self.sigmoid(y + self.bias)
        elif self.activation_function == 'tanh':
            self.output = self.tanh(y + self.bias)
        else:
            self.output = self.step(y + self.bias)
        return self.output

    def train(self, training_inputs, desired_output):
        self.list_w = np.zeros(training_inputs.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            for inp, output in zip(training_inputs, desired_output):
                real_output = self.feed(inp)
                update = self.learning_rate * (output - real_output)
                self.list_w += update * inp
                self.bias += update

    @property
    def step(self, y):
        return 0 if y < 0 else 1

    @property
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    @property
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class NeuronLayer:
    def __init__(self, layer_size, input_size, list_w, bias, learning_rate=0.1, epochs=100,
                 activation_function='sigmoid'):
        self.layer_size = layer_size
        self.input_size = input_size
        self.list_w = list_w
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function
        self.delta = 0
        self.perceptrons = []
        for _ in range(0, self.layer_size):
            self.perceptrons.append(Perceptron(list_w, bias, learning_rate, epochs, activation_function))

    def feed(self, inputs):
        layer_result = []
        for perceptron in self.perceptrons:
            layer_result.append(perceptron.feed(inputs))
        return layer_result

    def get_output(self):
        output = []
        for perceptron in self.perceptrons:
            output.append(perceptron.output)
        return output

    def get_weights(self):
        weights = []
        for perceptron in self.perceptrons:
            weights.append(perceptron.list_w)
        return weights


class NeuralNetwork:
    def __init__(self, amount_layers, list_perceptron_per_layer, input_size, output_size, list_w, bias,
                 learning_rate=0.1, epochs=100, activation_function='sigmoid'):
        self.amount_layers = amount_layers
        self.list_perceptron_per_layer = list_perceptron_per_layer
        self.input_size = input_size
        self.output_size = output_size
        self.list_w = list_w
        self.bias = bias
        self.learning_rate = learning_rate  # eta
        self.epochs = epochs
        self.activation_function = activation_function
        self.layers = []
        first_layer = NeuronLayer(list_perceptron_per_layer[0], input_size, list_w,
                                  bias, learning_rate, epochs, activation_function)
        self.layers.append(first_layer)
        for i in range(1, amount_layers):
            layer = NeuronLayer(list_perceptron_per_layer[i], list_perceptron_per_layer[i - 1], list_w,
                                bias, learning_rate, epochs, activation_function)
            self.layers.append(layer)

    def feed(self, inputs):
        for i in range(0, self.amount_layers):
            inputs = self.layers[i].feed(inputs)
        return inputs

    def train(self, inp, out):
        self.feed(inp)
        self.back_propagation(out)
        self.update_weights(inp)

    def derivative(self, output):
        if self.activation_function == 'sigmoid':
            return sigmoid_derivative(output)
        elif self.activation_function == 'tanh':
            return tanh_derivative(output)
        else:
            return 0

    def back_propagation(self, output):
        neuron_output = self.layers[-1].get_output()
        error = list(np.array(output) - np.array(neuron_output))
        delta = []
        for k in range(0, self.layers[-1].layer_size):
            delta.append(error[k] * self.derivative(neuron_output[k]))
        self.layers[-1].delta = delta
        for i in range(-1, -self.amount_layers, -1):
            layer_weights = self.layers[i].get_weights()
            for perceptron_weights in layer_weights:
                error = 0
                for weight in perceptron_weights:
                    error += weight * self.layers[i].perceptrons[neuron].delta

            for weight in range(0, len(layer_weights[0])):
                error = 0
                for neuron in range(0, self.layers[i].layer_size):
                    error += layer_weights[neuron][weight] * self.layers[i].perceptrons[neuron].delta
                delta = error * self.derivative(self.layers[i - 1].perceptrons[weight].output)
                self.layers[i - 1].perceptrons[weight].delta = delta

    def update_weights(self, inputs):
        previous_layer = []
        for layer in range(0, self.amount_layers - 1):
            for perceptron in self.layers[layer].perceptrons:
                perceptron.list_w = perceptron.list_w + self.learning_rate * perceptron.delta * inputs
                perceptron.bias = perceptron.bias + self.learning_rate * perceptron.delta
                previous_layer.append(perceptron.feed(inputs))
            inputs = previous_layer
            previous_layer = []
        return inputs

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.get_weights())
        return weights

    def error(self, training_input, training_output, testing_input, testing_output):
        original_weights = copy.deepcopy(self.get_weights())
        delta = []
        flag = True
        first_epoch = self.epochs
        n = len(training_output)
        n_test = len(testing_output)
        if self.output_size == 1:
            for i in range(0, n):
                training_output[i] = [training_output[i]]
                testing_output[i] = [testing_output[i]]
        errors = []
        precisions = []
        for i in range(0, self.epochs):
            for j in range(0, n):
                self.train(training_input[j], training_output[j])
            error = 0
            precision = 0
            for k in range(0, n_test):
                if np.argmax(self.feed(testing_input[k])) == np.argmax(testing_output[k]):
                    precision += 1
                for h in range(0, self.output_size):
                    error += abs(testing_output[k][h] - self.feed(testing_input[k])[h]) ** 2  # MSE
            errors.append(error)
            precisions.append(precision / n_test)
            if (precision / n_test) > 0.95 and flag:
                flag = False
                first_epoch = i
        last_weights = copy.deepcopy(self.get_weights())
        for i in range(0, len(original_weights)):
            delta.append(np.subtract(last_weights[i], original_weights[i]))
        return errors, precisions, first_epoch, delta


def normalize_breast_cancer(data, target):
    for sample in range(0, len(data[:])):
        data[:, sample] = data[:, sample] / np.max(data[:, sample])
    list_data = data.tolist()
    list_target = target.tolist()
    return list_data, list_target


def one_hot_encoding(samples, target):
    identity_matrix = np.eye(samples, dtype=int).tolist()
    for i in range(0, len(target)):
        target[i] = identity_matrix[target[i]]
    return target


def kfold_cross_validation(data, target):
    kf = KFold(n_splits=5, shuffle=True)
    fold_index = 1
    for train_index, test_index in kf.split(data):
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        listw = np.random.uniform(-2.0, 2.0, size=(1, len(data_train))).tolist()
        network = NeuralNetwork(amount_layers=5, list_perceptron_per_layer=[2, 4, 8, 4, 2],
                                input_size=len(data_train), output_size=len(target_train),
                                list_w=listw, bias=np.random.uniform(-2.0, 2-0))
        start_time = time.time()
        error, precision, epoch, weights = network.error(data_train, target_train, data_test, target_test)
        end_time = time.time()
        elapsed_time = end_time-start_time
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(error, range(0, network.epochs), color='tab:blue')
        ax.plot(precision, range(0, network.epochs), color='tab:orange')
        ax.set_title('fold num: ' + str(fold_index) + ', elapsed time: ' + str(elapsed_time))
        plt.show()
        fold_index = fold_index+1


cancer = load_breast_cancer()
data, target = normalize_breast_cancer(cancer.data, cancer.target)
target = one_hot_encoding(len(cancer.data[:]), target)

# 20% Test 80% Train

