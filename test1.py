import math
import random
import string

random.seed(0)

#generate a random number in interval [a,b)
def rand(a,b):
    return (b-a)*random.random()+a

#generate size I*J matrix,the default is a zero matrix(of course,you can also use Numpy speed)
def makem_atrix(m,n,fill=0.0):
    mat=[]
    for i in range(m):
        mat.append([fill]*n)
    return mat

#define sigmoid function
def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

#define the derivation of sigmoid function
def sigmoid_derivate(x):
    return x*(1-x)


