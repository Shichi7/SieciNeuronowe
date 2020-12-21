import idx2numpy
import cv2 as cv
from mnist import MNIST
import random
import numpy as np


class Loader:

    MNIST_DATA_PATH = "data"

    def __init__(self):
        self.train_inputs = None
        self.train_outputs = None
        self.test_inputs = None
        self.test_outputs = None
        self.train_inputs_raw = None
        self.train_outputs_raw = None
        self.test_inputs_raw = None
        self.test_outputs_raw = None
        self.loaded = False

        self.vector_len = 0
        pass

    def load_3D(self, log=False, preview=False):

        self.train_inputs = idx2numpy.convert_from_file("data/train/data1.idx3-ubyte")/255
        self.train_outputs = idx2numpy.convert_from_file("data/train/data1.idx1-ubyte")
        self.test_inputs = idx2numpy.convert_from_file("data/test/data1.idx3-ubyte")/255
        self.test_outputs = idx2numpy.convert_from_file("data/test/data1.idx1-ubyte")

        if all(loaded is not None for loaded in [self.train_inputs, self.train_outputs, self.test_inputs, self.test_outputs]):
            self.loaded = True

        if log:
            self.loader_log()

        if preview:
            self.preview_loaded_image()

    def preview_loaded_image(self):
        if self.loaded:
            index = random.randrange(0, len(self.train_inputs))

            #print(self.train_inputs[index].shape)

            cv.imshow("image", self.train_inputs[index])
            cv.waitKey(0)
            pass

    def load_with_mnist(self, log=False, preview=False):

        mnist_data = MNIST(self.MNIST_DATA_PATH)
        self.train_inputs_raw, self.train_outputs_raw = mnist_data.load_training()
        self.test_inputs_raw, self.test_outputs_raw = mnist_data.load_testing()

        if all(loaded is not None for loaded in [self.train_inputs_raw, self.train_outputs_raw, self.test_inputs_raw, self.test_outputs_raw]):
            self.vector_len = len(self.train_inputs_raw[0])
            self.convert_all()
            self.loaded = True

        if log:
            self.loader_log()

        if preview:
            self.preview_loaded_image_with_mnist(mnist_data)
        pass

    def convert_all(self):
        temp_train_inputs = np.asmatrix(np.where(np.array(self.train_inputs_raw) > 0.0, 1.0, 0.0))
        temp_test_inputs = np.asmatrix(np.where(np.array(self.test_inputs_raw) > 0.0, 1.0, 0.0))
        self.train_outputs = np.array(self.train_outputs_raw)
        self.test_outputs = np.array(self.test_outputs_raw)
        self.train_inputs = []
        self.test_inputs = []

        for train_input in temp_train_inputs:
            self.train_inputs.append(np.transpose(train_input))

        for test_input in temp_test_inputs:
            self.test_inputs.append(np.transpose(test_input))

        pass

    def preview_loaded_image_with_mnist(self, mnist_data):
        if self.loaded:
            index = random.randrange(0, len(self.train_inputs_raw))
            print(mnist_data.display(self.train_inputs_raw[index]))
        pass

    def loader_log(self):
        if self.loaded:
            comm = "Pomyślnie wczytano\n["+str(len(self.train_inputs))+"] wektorów treningowych\n"
            comm += "["+str(len(self.test_inputs))+"] wektorów testowych\n"
        else:
            comm = "Błąd wczytywania\n"

        print(comm)
        pass

