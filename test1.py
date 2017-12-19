import math
import random
import string

random.seed(0)

#generate a random number in interval [a,b)
def rand(a,b):
    return (b-a)*random.random()+a

#generate size I*J matrix,the default is a zero matrix(of course,you can also use Numpy speed)
def makeMatrix(I,J,fill=0):
    m=[]
    for i in range(I):
        m.append([fill]*J)
    return m

