# coding:utf-8
import numpy as np


def sgn(a):     #sgn(x)
    if a > 0:
        return 1
    elif a <= 0:
        return -1
    else:
        print("sgn error")
        return 0


#the Class used to save sample array
class TrainUnit:
    z = np.zeros(1, int)    # input layer array
    y = np.zeros(1, int)    # hidden layer array
    o = np.zeros(1, int)    # output layer array
    d = np.zeros(1, int)    # ideal output layer array according the classnum
    classnum = 0            # belong to which class. take 0 as unknown
    result = 0              # the result of recognize
    I = 0
    J = 0
    K = 0

    def __init__(self, I, K, J):
        self.I = I
        self.K = K
        self.J = J

    # set the three layers arrays
    def set(self, yary, classnum):

        z2add = yary.reshape(-1, 1)
        z2add = np.vstack((z2add, np.array([-1])))
        self.z = z2add

        y2add = np.zeros((self.J, 1), int)
        y2add[self.J - 1] = -1
        self.y = y2add

        self.classnum = classnum
        d2add = np.zeros((self.K, 1), int)
        d2add += -1
        o2add = d2add.copy()
        d2add[classnum - 1] = 1
        self.d = d2add
        self.o = o2add

    def sety(self, y):  # set the last element of y at -1
        self.y = y
        self.y[self.J - 1, 0] = -1

    def seto(self, o):
        self.o = o

    # recognize the sample. the index of the biggest number in o is the classnum.
    def calresult(self):
        self.result = 0
        Max = 0
        for i in range(self.K):
            if self.o[i, 0] > Max:
                self.result = i + 1
                Max = self.o[i, 0]
        return self.result
