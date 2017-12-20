import numpy as np

class Network(object):

    def __init__(self,sizes):
        '''

        :param sizes: list type,save the number of neurons in every layer.
        for example:sizes[2,3,2] represent a network with 2 neurons in input layer,3 neurons
        in hidden layer and 2 neurons in output layer

        '''

        #the number of layers
        self.num_layers=len(sizes)
        self.sizes=sizes

        # generate biases between (0,1) for y neurons in every layer except input layer
        self.biases=[np.random.randn(y,1) for y in sizes[1,:]]

        #generate the weights between (0,1) for every connecting line
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]