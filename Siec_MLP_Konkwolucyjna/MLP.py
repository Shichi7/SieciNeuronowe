from MLPLayer import MLPLayer
from ConvoLayer import ConvoLayer
import numpy as np


class MLP:
    BATCH_SIZE = 25
    NUMBER_OF_CLASSES = 10
    HIDDEN_LAYERS_DEF = [20]

    def __init__(self, loader):
        self.loader = loader

        self.layers = []
        self.correct_count = 0

        for layer_number in range(len(self.HIDDEN_LAYERS_DEF)):
            if layer_number == 0:
                previous_neuron_count = self.loader.vector_len
            else:
                previous_neuron_count = self.layers[layer_number-1].neuron_count

            new_layer = MLPLayer(self.HIDDEN_LAYERS_DEF[layer_number], previous_neuron_count)
            self.layers.append(new_layer)

        if len(self.HIDDEN_LAYERS_DEF) > 0:
            last_hidden_neuron_count = self.HIDDEN_LAYERS_DEF[-1]
        else:
            last_hidden_neuron_count = self.loader.vector_len

        self.layers.append(MLPLayer(self.NUMBER_OF_CLASSES, last_hidden_neuron_count, True))

        pass

    def train(self, iterations=10, test_flag=False, skip_flag=False, confusion_flag=False):
        data_x = self.loader.train_inputs
        data_y = self.loader.train_outputs

        for i in range(iterations):
            mean_lms = 0
            batch_count = 1
            vector_count = len(data_x)
            for vector_index in range(vector_count):
                self.pass_forward(data_x[vector_index])
                self.layers[-1].output_layer_error(data_y[vector_index])
                self.project_back()
                batch_count = self.weights_update(batch_count, vector_index, vector_count)
                mean_lms += self.layers[-1].square_error
            mean_lms /= vector_count
            print("ITERACJA: ["+str(i)+"] LMS: ["+str(mean_lms)+"]")

            if confusion_flag:
                self.confusion_matrix()

            if test_flag:
                self.test(i, skip_flag)
        pass

    def pass_forward(self, current_vector_x):
        for layer_number in range(len(self.layers)):
            if layer_number == 0:
                previous_output = current_vector_x
            else:
                previous_output = self.layers[layer_number - 1].output_vector

            self.layers[layer_number].predict_output(previous_output)
        pass

    def project_back(self):
        for layer_number in range(len(self.layers) - 1):
            index = len(self.layers) - 2 - layer_number
            next_error = self.layers[index + 1].error
            next_weights = self.layers[index + 1].weights
            self.layers[index].project_error(next_error, next_weights)
        pass

    def weights_update(self, batch_count, vector_index, x_len):
        alt_count = x_len-1
        if (batch_count == self.BATCH_SIZE) or (vector_index == alt_count):
            batch_count = 1
            for layer_number in range(len(self.layers)):
                self.layers[layer_number].weights_update()
        else:
            batch_count += 1

        return batch_count

    def test(self, iteration, skip_flag):
        self.correct_count = 0
        data_x = self.loader.test_inputs
        data_y = self.loader.test_outputs
        vector_count = len(data_x)
        for vector_index in range(vector_count):
            self.pass_forward(data_x[vector_index])
            if self.layers[-1].output == data_y[vector_index]:
                self.correct_count += 1

        if skip_flag:
            if (iteration + 1) % 2 == 0:
                print(str(self.correct_count) + "/" + str(len(self.loader.test_inputs)))
                print(str(self.correct_count / len(self.loader.test_inputs) * 100) + "%")
        else:
            print(str(self.correct_count) + "/" + str(len(self.loader.test_inputs)))
            print(str(self.correct_count / len(self.loader.test_inputs) * 100) + "%")

        pass

    def confusion_matrix(self):
        confusion_data = np.zeros((10, 10), dtype=int)

        self.correct_count = 0
        data_x = self.loader.test_inputs
        data_y = self.loader.test_outputs
        vector_count = len(data_x)
        for vector_index in range(vector_count):
            self.pass_forward(data_x[vector_index])
            confusion_data[data_y[vector_index]][self.layers[-1].output]+=1

        print(confusion_data)
        pass



