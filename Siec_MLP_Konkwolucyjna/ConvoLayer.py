import numpy as np
from ActivationFuncEnum import Activations
from OptimizerEnum import Optimizers
from OptimizerEnum import WeightOptimizers
from scipy.special import softmax
import math
import skimage.measure


class ConvoLayer:
    DEF_ACTIVATION = Activations.relu
    WEIGHTS_OPTIMIZER = WeightOptimizers.HE  # /he/xavier/NONE/
    DEF_STANDARD_DEV = 0.1

    def __init__(self, prev_count, prev_size):

        self.activation_func = self.DEF_ACTIVATION

        self.flattened_output = None
        self.pooling_outputs = []
        self.outputs = []
        self.outputs2 = []
        self.filters = []
        self.biases = []

        self.prev_output = None
        self.prev_count = prev_count
        self.prev_size = prev_size
        self.results_count = 32

        self.step = 1
        self.filter_size = 3
        self.padding = 1

        if self.WEIGHTS_OPTIMIZER == WeightOptimizers.HE:
            standard_dev = math.sqrt(2.0 / self.prev_count)
        elif self.WEIGHTS_OPTIMIZER == WeightOptimizers.XAVIER:
            standard_dev = math.sqrt(2.0 / (self.prev_count + self.results_count))
        else:
            standard_dev = self.DEF_STANDARD_DEV

        self.result_size = int((self.prev_size - self.filter_size + 2 * self.padding) / self.step + 1)
        self.pooling_size = int(self.result_size/2)

        self.filters = []

        for i in range(self.results_count):
            self.pooling_outputs.append(np.zeros((self.pooling_size, self.pooling_size)))
            self.outputs.append(np.zeros((self.result_size, self.result_size)))
            self.filters.append(np.random.normal(0.0, standard_dev, (self.filter_size, self.filter_size)))
            self.biases.append(np.random.normal(0.0, standard_dev, (self.result_size, self.result_size)))
        pass

    def predict_output(self, previous_layer_output):

        self.add_padding(previous_layer_output)

        for i in range(self.results_count):
            for x in range(self.result_size):
                for y in range(self.result_size):
                    self.convolution(i, x, y)

            self.outputs[i] += self.biases[i]
            self.calc_activation(i, self.outputs[i])

        input("after convo shape")
        print(self.outputs[0].shape)

        pass

    def max_pooling(self):
        for i in range(self.results_count):
            self.pooling_outputs[i] = skimage.measure.block_reduce(self.outputs[i], (2, 2), np.max)

        input("after pooling shape")
        print(self.pooling_outputs[0].shape)

    def convolution(self, curr_output, curr_x, curr_y):
        sum = 0.0

        for a in range(self.filter_size):
            for b in range(self.filter_size):
                sum += self.filters[curr_output][a, b] * self.prev_output[curr_x+a, curr_y+b]

        self.outputs[curr_output][curr_x, curr_y] = sum
        pass

    def flatten(self):
        self.flattened_output = []

        for i in range(self.results_count):
            self.pooling_outputs[i] = self.pooling_outputs[i].flatten()
            self.flattened_output = np.append(self.flattened_output, self.pooling_outputs[i])

        self.flattened_output = np.asmatrix(self.flattened_output).transpose()
        input("after flattening shape")
        print(self.flattened_output.shape)

        pass

    def add_padding(self, prev_output):
        self.prev_output = np.zeros((self.prev_size+2*self.padding, self.prev_size+2*self.padding))
        self.prev_output[self.padding:self.prev_size+self.padding, self.padding:self.prev_size+self.padding] = prev_output

        pass

    def calc_activation(self, curr_output, excitation):

        if self.activation_func == Activations.tanh:
            self.outputs[curr_output] = np.tanh(excitation)
        elif self.activation_func == Activations.relu:
            self.outputs[curr_output] = np.vectorize(self.relu_activation, otypes=[float])(excitation)
        pass

    def relu_activation(self, x):
        return 0 if x < 0 else x