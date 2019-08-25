import numpy as np


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
        for i in range(0, self.layer_size):
            layer_result.append(self.perceptrons[i].feed(inputs))
        return layer_result

    def get_output(self):
        output = []
        for i in range(0, self.layer_size):
            output.append(self.perceptrons[i].output)
        return output


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

    def train(self, inp):
        output = self.feed(inp)
        self.back_propagation(output)
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
            weights = self.layers[i].list_w
            for weight in range(0, len(weights[0])):
                error = 0
                for neuron in range(0, self.layers[i].layer_size):
                    error += weights[neuron][weight] * self.layers[i].perceptrons[neuron].delta
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
