import math
import random
import string

random.seed(0)


# generate a random number in interval [a,b)
def rand(a, b):
    return (b - a) * random.random() + a


# generate size I*J matrix,the default is a zero matrix(of course,you can also use Numpy speed)
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


# define sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# define the derivation of sigmoid function
def sigmoid_derivate(x):
    return x * (1 - x)


class BP_network(object):
    def __init__(self):
        self.input_n = 0  # the number of neurons in input layer
        self.hidden_n = 0  # the number of neurons in hidden layer
        self.output_n = 0  # the number of neurons in output layer
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        '''
        :param ni: the number of neurons in input layer
        :param nh: the number of neurons in hidden layer
        :param no: the number of neurons in output layer
        :return:
        '''

        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no

        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n

        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)

        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-2.0, 2.0)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)

        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        # return self.output_cells[:]
        return self.output_cells

    def back_propagate(self, case, label, learn, correct):
        '''

        :param case: training sample
        :param label:
        :param learn: rate
        :param correct:
        :return:
        '''

        # feed forward
        self.predict(case)

        # get outputlayer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivate(self.output_cells[o]) * error
