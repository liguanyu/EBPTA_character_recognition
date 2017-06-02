# coding:utf-8
import numpy as np
from TrainUnit import *


# the EBPTA algorithm in book "introduction to Artificial Neural"(Jacek M. Zurada), at Page 188

# f(x)
def ff(a):
    return 2 / (1 + np.exp(-a)) - 1


# f(x) to matrix
def f(aary):
    if type(aary) == np.ndarray:  # for each element x in the matrix, do ff(x)
        temp = aary.copy()
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                temp[i, j] = ff(temp[i, j])
        return temp
    else:
        return ff(aary)


# EBPTA algorithm
class myEBPTA:
    eta = 0.3  # iteration coefficient
    E = 0  # error
    P = 0  # the length of training set
    J = 0  # hidden layer
    K = 0  # output layer
    I = 0  # input layer
    w = np.zeros(1)  # w matrix
    v = np.zeros(1)  # v matrix
    unitlist = []  # training set
    delta_o = np.zeros(1)  # delta
    delta_y = np.zeros(1)
    EMAX = 2  # error MAX
    cntMAX = 1000  # max iteration times
    cnt = 0  # iteration times

    # initial the length of the three layers
    def __init__(self, sample_length, output_length, hidden_length):
        self.K = output_length
        self.I = sample_length + 1  # the length of input layer is one bigger the that of sample,
        # as the last parameter of input is -1
        self.J = hidden_length
        print("begin")

    # add training set , argumengt as (sample1 array1, the class of sample1, array2, class2, ...)
    def addtrainset(self, *args):
        if len(args) % 2 != 0:
            print("addTrainSet error")
            return 0

        for i in range(len(args)):  # if the length of argument is even
            if i % 2 == 0:  # use Class TrainUnit to save sample array
                unit = TrainUnit(self.I, self.K, self.J)
                unit.set(args[i], args[i + 1])
                self.unitlist.append(unit)
                self.P += 1

        return len(self.unitlist)

    def setEMAX(self, EMAX):
        self.EMAX = EMAX

    def setcntMAX(self, cntMAX):
        self.cntMAX = cntMAX

    def seteta(self, eta):
        self.eta = eta

    # initial w matrix, v matrix
    def initProcess(self):
        self.w = (np.random.rand(self.K, self.J)) * 2 - 1
        print(self.w)
        self.v = (np.random.rand(self.J, self.I)) * 2 - 1
        print(self.v)
        self.cnt = 0

    # start the train
    def start(self):
        self.E = self.EMAX + 1  # begin the first train
        while self.E > self.EMAX and self.cnt < self.cntMAX:  # if error > EMAX or reach the max iteration,
                                                                # then end the train
            self.E = 0
            self.cnt += 1
            self.train()
            print(self.E)
        print("%d trains have been made" % (self.cnt))

    def train(self):       # train
        print("the NO.%d train" % (self.cnt))
        for p in range(self.P):     # according to the book
            self.caly(p)
            self.calo(p)
            self.calE(p)
            self.caldelta(p)
            self.OutputLayerAdjust(p)
            self.HiddenLayerAdjust(p)



    def caly(self, p):
        self.unitlist[p].sety(f(np.dot(self.v, self.unitlist[p].z)))

    def calo(self, p):
        self.unitlist[p].seto(f(np.dot(self.w, self.unitlist[p].y)))

    def calE(self, p):
        temp = self.unitlist[p].d - self.unitlist[p].o
        self.E += 0.5 * (temp * temp).sum()

    def caldelta(self, p):
        d = self.unitlist[p].d
        o = self.unitlist[p].o
        y = self.unitlist[p].y

        self.delta_o = 0.5 * (d - o) * (1 - o * o)

        self.delta_y = np.zeros((self.J, 1))
        for j in range(self.J):
            temp = 0
            for k in range(self.K):
                temp += self.delta_o[k, 0] * self.w[k, j]
            self.delta_y[j, 0] = 0.5 * (1 - y[j, 0] * y[j, 0]) * temp

    def OutputLayerAdjust(self, p):
        self.w = self.w + self.eta * np.dot(self.delta_o, self.unitlist[p].y.transpose())

    def HiddenLayerAdjust(self, p):
        self.v = self.v + self.eta * np.dot(self.delta_y, self.unitlist[p].z.transpose())

    # recognize a sample by calculation the output layer
    def recognize(self, sample):
        unit = TrainUnit(self.I, self.K, self.J)
        unit.set(sample, 0)
        unit.sety(f(np.dot(self.v, unit.z)))
        unit.seto(f(np.dot(self.w, unit.y)))
        result = unit.calresult()
        del unit
        return result

