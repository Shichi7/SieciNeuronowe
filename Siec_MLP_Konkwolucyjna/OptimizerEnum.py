from enum import Enum

class Optimizers(Enum):
    SIMPLE_MOMENTUM = 1
    NESTEROV = 2
    ADAGARD = 3
    ADADELTA = 4
    ADAM = 5
    NONE = 6

class WeightOptimizers(Enum):
    XAVIER = 1
    HE = 2
    NONE = 3

