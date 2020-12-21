from MLPLayer import MLPLayer
from ConvoLayer import ConvoLayer
import numpy as np


class Convo:
    NUMBER_OF_CLASSES = 10
    JOINED_NEURONS_COUNT = 20

    def __init__(self, loader):
        self.loader = loader
        self.convo_layer = ConvoLayer(1, 28)
        self.joined_layer = MLPLayer(self.JOINED_NEURONS_COUNT, 6272)
        self.softmax_layer = MLPLayer(10, self.JOINED_NEURONS_COUNT, True)
        pass

    def train(self, iterations=10, test_flag=True):
        data_x = self.loader.train_inputs
        data_y = self.loader.train_outputs

        for i in range(iterations):
            mean_lms = 0
            vector_count = len(data_x)
            for vector_index in range(vector_count):
                self.pass_forward(data_x[vector_index])
                self.softmax_layer.output_layer_error(data_y[vector_index])

                mean_lms += self.softmax_layer.square_error
            mean_lms /= vector_count
            print("ITERACJA: ["+str(i)+"] LMS: ["+str(mean_lms)+"]")

        pass

    def pass_forward(self, current_vector_x):

        self.convo_layer.predict_output(current_vector_x)
        self.convo_layer.max_pooling()
        self.convo_layer.flatten()

        self.joined_layer.predict_output(self.convo_layer.flattened_output)
        self.softmax_layer.predict_output(self.joined_layer.output_vector)

        pass


