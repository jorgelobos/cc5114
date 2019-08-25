import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from math import floor
from math import ceil

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.errors = []
        self.learning_rate = learning_rate  # eta
        self.epochs = epochs

    def learning_algorithm(self, training_inputs, desired_output):
        self.listw = np.zeros(training_inputs.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            errors = 0
            for input, output in zip(training_inputs, desired_output):
                real_output = Logic(self, input)
                update = self.learning_rate * (output - real_output)
                self.listw += update * input
                self.bias += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self


def Logic(perceptron, inp):
    x = np.array(inp)
    w = np.array(perceptron.listw)
    y = np.sum(w * x)
    if y + perceptron.bias <= 0:
        return 0
    else:
        return 1


def runnable():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # setosa and versicolor
    desired_output = df.iloc[0:100, 4].values
    desired_output = np.where(desired_output == 'Iris-setosa', -1, 1)

    # sepal length and petal length
    training_inputs = df.iloc[0:100, [0, 2]].values

    ppn = Perceptron()
    ppn.learning_algorithm(training_inputs, desired_output)
    plot_decision_regions(training_inputs, desired_output, clf=ppn)
    plt.plot()
    print('Weights: %s' % ppn.listw)
    plt.title('Perceptron')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.show()

    plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Misclassifications')
    plt.show()

def plot_decision_regions(X, y, clf,
                          zoom_factor=1.,
                          legend=1,
                          hide_spines=True,
                          markers='s^oxv<>',
                          colors=('#1f77b4,#ff7f0e,#3ca02c,#d62728,'
                                  '#9467bd,#8c564b,#e377c2,'
                                  '#7f7f7f,#bcbd22,#17becf'),
                          scatter_kwargs=None):

    dim = X.shape[1]
    ax = plt.gca()

    # Extra input validation for higher number of training features

    marker_gen = cycle(list(markers))

    n_classes = np.unique(y).shape[0]
    colors = colors.split(',')
    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]

    feature_index = (0, 1)
    x_index, y_index = feature_index

    # Get minimum and maximum
    x_min, x_max = (X[:, x_index].min() - 1./zoom_factor,
                    X[:, x_index].max() + 1./zoom_factor)

    y_min, y_max = (X[:, y_index].min() - 1./zoom_factor,
                    X[:, y_index].max() + 1./zoom_factor)

    xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
    xnum, ynum = floor(xnum), ceil(ynum)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=xnum),
                         np.linspace(y_min, y_max, num=ynum))

    X_grid = np.array([xx.ravel(), yy.ravel()]).T
    X_predict = np.zeros((X_grid.shape[0], dim))
    X_predict[:, x_index] = X_grid[:, 0]
    X_predict[:, y_index] = X_grid[:, 1]

    ax.axis(xmin=xx.min(), xmax=xx.max(), y_min=yy.min(), y_max=yy.max())

    for idx, c in enumerate(np.unique(y)):
        y_data = X[y == c, y_index]
        x_data = X[y == c, x_index]
        ax.scatter(x=x_data,
                   y=y_data,
                   c=colors[idx],
                   marker=next(marker_gen),
                   label=c)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, framealpha=0.3, scatterpoints=1, loc=legend)
    ax.plot(r0 * np.cos(theta), r0 * np.sin(theta))
    return ax


if __name__ == '__main__':
    runnable()
    # unittest.main()
