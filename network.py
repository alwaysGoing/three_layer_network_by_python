import numpy as np
import random

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

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

    def feedforward(self,a):
        '''

        :param a:input
        :return: the result of every neuron after compution

        '''

        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        '''

        :param training_data:
        :param epochs:
        :param mini_batch_size:
        :param eta: learning rate
        :param test_data:
        :return:
        '''

        if test_data:
            n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            #disrupt the training set to change the sort order
            random.shuffle(training_data)
            #the training set is divided to the mini_batch
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("epoch{0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("epoch{0} complete".format(j))

    def update_mini_batch(self,mini_batch,eta):
        '''

        :param mini_batch:
        :param eta:
        :return:
        '''

        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b,delta_nable_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nable_w)]

        self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]

    def backprop(self,x,y):
        '''

        :param x:trainging samples
        :param y: labels
        :return:
        '''

        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        activation=x
        activations=[x]
        zs=[]
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)

        delta=self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())
        for l in range(2,self.num_layers):
            z=zs[-l]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-1]=delta
            nabla_w[-1]=np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    def evaluate(self,test_data):
        test_result=[(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_result)

