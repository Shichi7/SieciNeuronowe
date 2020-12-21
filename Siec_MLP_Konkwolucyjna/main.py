from Loader import Loader
from MLP import MLP
from Convo import Convo


def run_MLP():
    loader = Loader()
    loader.load_with_mnist(True, False)
    MLP_network = MLP(loader)
    MLP_network.train(50, True, False)

    pass


def run_Convo():

    loader = Loader()
    loader.load_3D(True, False)

    ConvoNetwork = Convo(loader)
    ConvoNetwork.train(10)

    pass


if __name__ == '__main__':
    #run_MLP()
    run_Convo()

    pass


