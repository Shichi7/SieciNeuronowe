import numpy as np
from ActivationFuncEnum import Activations
from OptimizerEnum import Optimizers
from OptimizerEnum import WeightOptimizers
from scipy.special import softmax
import math


class MLPLayer:

    DEF_ACTIVATION = Activations.tanh
    OPTIMIZER = Optimizers.ADADELTA   #/simple_momentum/nesterov/adagard/adadelta/adam/NONE/
    WEIGHTS_OPTIMIZER = WeightOptimizers.HE   #/he/xavier/NONE/

    DEF_ALPHA_FACTOR = 0.001
    DEF_STANDARD_DEV = 0.1

    def __init__(self, neuron_count, prev_neuron_count, output_layer=False):

        self.activation_func = self.DEF_ACTIVATION
        self.alpha_factor = self.DEF_ALPHA_FACTOR
        self.output = -1

        self.square_error = float('inf')

        self.weight_modifiers_history = []
        self.bias_modifiers_history = []

        self.nesterov_output_vector = None
        self.output_vector = None
        self.error = None
        self.previous_layer_output = None
        self.activation_gradient = None

        if self.OPTIMIZER == Optimizers.ADAGARD:
            self.adagrad_matrix_weights = np.zeros((neuron_count, prev_neuron_count))
            self.adagrad_matrix_biases = np.zeros((neuron_count, 1))

        if self.OPTIMIZER == Optimizers.ADADELTA:
            self.adadelta_matrix_weights = np.zeros((neuron_count, prev_neuron_count))
            self.adadelta_matrix_biases = np.zeros((neuron_count, 1))
            self.adadelta_mod_weights = np.zeros((neuron_count, prev_neuron_count))
            self.adadelta_mod_biases = np.zeros((neuron_count, 1))

        if self.OPTIMIZER == Optimizers.ADAM:
            self.adam_matrix_weights_square = np.zeros((neuron_count, prev_neuron_count))
            self.adam_matrix_biases_square = np.zeros((neuron_count, 1))
            self.adam_matrix_weights = np.zeros((neuron_count, prev_neuron_count))
            self.adam_matrix_biases = np.zeros((neuron_count, 1))

        self.output_layer = output_layer
        self.neuron_count = neuron_count

        self.prev_mod_weights = np.zeros((neuron_count, prev_neuron_count))
        self.prev_mod_biases = np.zeros((neuron_count, 1))

        if self.WEIGHTS_OPTIMIZER == WeightOptimizers.HE:
            standard_dev = math.sqrt(2.0/prev_neuron_count)
        elif self.WEIGHTS_OPTIMIZER == WeightOptimizers.XAVIER:
            standard_dev = math.sqrt(2.0/(prev_neuron_count + neuron_count))
        else:
            standard_dev = self.DEF_STANDARD_DEV

        self.weights = np.random.normal(0.0, standard_dev, (neuron_count, prev_neuron_count))
        self.biases = np.random.normal(0.0, standard_dev, (neuron_count, 1))

        pass

    def predict_output(self, previous_layer_output):
        self.previous_layer_output = previous_layer_output

        excitation = self.weights @ previous_layer_output + self.biases
        self.calc_activation(excitation)

        if self.OPTIMIZER == Optimizers.NESTEROV:
            self.calc_nesterov_params()
        else:
            self.calc_gradient_output(excitation, self.output_vector)

        if self.output_layer:
            self.output = np.argmax(self.output_vector)

        #self.debug_print(excitation, True)

        pass

    def calc_nesterov_params(self):
        momentum_weights = self.weights - 0.9 * self.prev_mod_weights
        momentum_biases = self.biases - 0.9 * self.prev_mod_biases
        nesterov_excitation = momentum_weights @ self.previous_layer_output + momentum_biases
        self.calc_activation(nesterov_excitation, True)
        self.calc_gradient_output(nesterov_excitation, self.nesterov_output_vector)
    pass

    def output_layer_error(self, expected_y):
        y = np.expand_dims(np.zeros(self.neuron_count), axis=1)
        y[expected_y] = 1

        error_vector = self.output_vector - y
        self.square_error = np.sum(np.power(error_vector, 2))/2

        if self.OPTIMIZER == Optimizers.NESTEROV:
            error_vector = self.nesterov_output_vector - y

        self.error = np.multiply(error_vector, self.activation_gradient)

        self.save_error_history(self.error)
        pass

    def project_error(self, next_errors, next_weights):

        self.error = np.multiply(np.transpose(next_weights) @ next_errors, self.activation_gradient)
        self.save_error_history(self.error)
        pass

    def save_error_history(self, error):
        weight_modifier = error @ np.transpose(self.previous_layer_output)

        self.bias_modifiers_history.append(error)
        self.weight_modifiers_history.append(weight_modifier)
        pass

    def weights_update(self):

        E_factor = 0.00000001
        batch_size = len(self.bias_modifiers_history)
        weight_error_mean = sum(self.weight_modifiers_history)/batch_size
        bias_error_mean = sum(self.bias_modifiers_history)/batch_size

        if self.OPTIMIZER == Optimizers.ADAGARD:
            self.adagrad_matrix_weights += np.square(weight_error_mean)
            self.adagrad_matrix_biases += np.square(bias_error_mean)

            current_mod_weights = np.multiply(self.alpha_factor/np.sqrt(self.adagrad_matrix_weights + E_factor), weight_error_mean)
            current_mod_biases = np.multiply(self.alpha_factor/np.sqrt(self.adagrad_matrix_biases + E_factor), bias_error_mean)

        elif self.OPTIMIZER == Optimizers.ADADELTA:
            self.adadelta_matrix_weights = 0.9 * self.adadelta_matrix_weights + 0.1 * np.square(weight_error_mean)
            self.adadelta_matrix_biases = 0.9 * self.adadelta_matrix_biases + 0.1 * np.square(bias_error_mean)

            current_mod_weights = np.multiply(np.sqrt(self.adadelta_mod_weights+E_factor)/np.sqrt(self.adadelta_matrix_weights + E_factor), weight_error_mean)
            current_mod_biases = np.multiply(np.sqrt(self.adadelta_mod_biases+E_factor)/np.sqrt(self.adadelta_matrix_biases + E_factor), bias_error_mean)

            self.adadelta_mod_weights = 0.9 * self.adadelta_mod_weights + 0.1 * np.square(current_mod_weights)
            self.adadelta_mod_biases = 0.9 * self.adadelta_mod_biases + 0.1 * np.square(current_mod_biases)

        elif self.OPTIMIZER == Optimizers.ADAM:

            self.adam_matrix_weights_square = 0.9 * self.adam_matrix_weights_square + 0.1 * np.square(weight_error_mean)
            self.adam_matrix_biases_square = 0.9 * self.adam_matrix_biases_square + 0.1 * np.square(bias_error_mean)

            self.adam_matrix_weights = 0.999 * self.adam_matrix_weights + 0.001 * weight_error_mean
            self.adam_matrix_biases = 0.999 * self.adam_matrix_biases + 0.001 * bias_error_mean

            m_weights = self.adam_matrix_weights_square/0.1
            m_biases = self.adam_matrix_biases_square/0.1

            v_weights = self.adam_matrix_weights/0.001
            v_biases = self.adam_matrix_biases/0.001

            current_mod_weights = np.multiply(self.alpha_factor/(np.sqrt(m_weights) + E_factor), v_weights)
            current_mod_biases = np.multiply(self.alpha_factor/(np.sqrt(m_biases) + E_factor), v_biases)

        elif self.OPTIMIZER == Optimizers.NONE:
            current_mod_weights = self.alpha_factor * weight_error_mean
            current_mod_biases = self.alpha_factor * bias_error_mean

        else:
            prev_momentum_weights = 0.9 * self.prev_mod_weights
            prev_momentum_biases = 0.9 * self.prev_mod_biases

            current_mod_weights = prev_momentum_weights + self.alpha_factor * weight_error_mean
            current_mod_biases = prev_momentum_biases + self.alpha_factor * bias_error_mean

            self.prev_mod_weights = current_mod_weights
            self.prev_mod_biases = current_mod_biases

        self.weights = self.weights - current_mod_weights
        self.biases = self.biases - current_mod_biases

        self.weight_modifiers_history = []
        self.bias_modifiers_history = []
        pass

    def calc_activation(self, excitation, nesterov_flag=False):

        if self.output_layer:
            output_vect = softmax(excitation)
        elif self.activation_func == Activations.tanh:
            output_vect = np.tanh(excitation)
        elif self.activation_func == Activations.relu:
            output_vect = np.vectorize(self.relu_activation, otypes=[float])(excitation)

        if nesterov_flag:
            self.nesterov_output_vector = output_vect
        else:
            self.output_vector = output_vect
        pass

    def calc_gradient_output(self, excitation, output):
        if self.output_layer:
            self.activation_gradient = np.vectorize(self.relu_softplus_gradient, otypes=[float])(excitation)
        elif self.activation_func == Activations.tanh:
            self.activation_gradient = np.vectorize(self.tanh_gradient, otypes=[float])(output)
        elif self.activation_func == Activations.relu:
            self.activation_gradient = np.vectorize(self.relu_softplus_gradient, otypes=[float])(excitation)
        pass

    def tanh_gradient(self, x):
        return 1 - math.pow(x, 2)

    def relu_activation(self, x):
        return 0 if x < 0 else x

    def relu_softplus_gradient(self, x):
        return 1/(1 + math.exp(-x))

    def print_weights(self):
        print(self.weights)
        pass

    def debug_print(self, activation, stop=False):
        if stop:
            input("wektor wejsciowy")
        print(self.previous_layer_output)
        if stop:
            input("wagi")
        print(self.weights)
        if stop:
            input("biasy")
        print(self.biases)
        if stop:
            input("output")
        print(activation)
        if stop:
            input("aktywacja")
        print(self.output_vector)
        if stop:
            input("gradient aktywacji")
        print(self.activation_gradient)
        if stop:
            input("KONIEC")
        pass



