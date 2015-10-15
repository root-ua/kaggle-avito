# This is an adaptive-learning-rate sparse logistic-regression with
# efficient L1-L2-regularization

from math import sqrt, exp, log
from csv import DictReader
import pandas as pd
import numpy as np


class Ftrl(object):
    def __init__(self, alpha, beta, l1, l2, bits):
        self.z = [0.] * bits # array of weights
        self.n = [0.] * bits # array of squared sums of past gradients
        self.alpha = alpha # learning rate
        self.beta = beta # smoothing parameter for adaptive learning rate
        self.l1 = l1 # L-1 regularization parameter, larger value means more regularized
        self.l2 = l2 # L-2 regularization parameter, larger value means more regularized
        self.w = {} # dictionary of lazy weights for each feature (using dictionary for memory usage reduction)
        self.X = [] # features
        self.y = 0. # actuall value
        self.bits = bits #
        self.Prediction = 0. # predicted value

    def sgn(self, x):
        if x < 0:
            return -1
        else:
            return 1

    def fit(self, line):
        try:
            self.ID = line['ID']
            del line['ID']
        except:
            pass

        try:
            self.y = float(line['IsClick'])
            del line['IsClick']
        except:
            pass

        del line['HistCTR']

        self.X = [0.] * len(line)

        for i, key in enumerate(line): # enumerate/sort is for preserving feature ordering
            val = line[key]
            # one-hot encode everything with hash trick
            self.X[i] = (abs(hash(key + '_' + val)) % self.bits)

        self.X = [0] + self.X
        print "X hash ", self.X, "\n"

    def logloss(self):
        act = self.y
        pred = self.Prediction
        predicted = max(min(pred, 1. - 10e-15), 10e-15)
        return -log(predicted) if act == 1. else -log(1. - predicted)

    def logloss_new(self):
        # logarithmic loss of p given y
        # p: our prediction
        # y: real answer
        y = self.y
        p = max(min(self.Prediction, 1. - 10e-15), 10e-15)
        return -(y*log(p) + (1-y)*log(1-p))

    def predict(self):
        W_dot_x = 0.
        w = {}
        for i in self.X:
            if abs(self.z[i]) <= self.l1:
                w[i] = 0.
                print "feature = ", i, "l1 = ", self.l1, "z = ", self.z[i], "w = ", w[i], "\n"
            else:
                w[i] = (self.sgn(self.z[i]) * self.l1 - self.z[i]) / (
                    ((self.beta + sqrt(self.n[i])) / self.alpha) + self.l2)

                print "weight update: ", "(sgn(", self.z[i], ") * ", self.l1, " - ", self.z[i], ") / ((( ", self.beta, " + sqrt( ", self.n[i], ")) / ", self.alpha, " ) + ", self.l2, " )"
                print "feature = ", i, "l1 = ", self.l1, "z = ", self.z[i], "w = ", w[i], "\n"

            W_dot_x += w[i]
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        self.Prediction = 1. / (1. + exp(-max(min(W_dot_x, 35.), -35.)))

        print "Min: ", min(W_dot_x, 35.)
        print "Max: ", -max(min(W_dot_x, 35.), -35.0)
        print "Exp: ", exp(-max(min(W_dot_x, 35.), -35.0))
        print "Prediction rule = 1 / (1 + exp( -max( min(", W_dot_x, "), 35), -35)))"

        return self.Prediction

    def update(self, prediction):
        for i in self.X:

            # gradient under logloss
            g = (prediction - self.y)  # * i

            sigma = (1. / self.alpha) * (sqrt(self.n[i] + g * g) - sqrt(self.n[i]))
            self.z[i] += g - sigma * self.w[i]
            self.n[i] += g * g


if __name__ == '__main__':

    """
    SearchID	AdID	Position	ObjectType	HistCTR	IsClick
    """
    train = 'data/trainSearchStream.tsv'
    clf = Ftrl(alpha=0.1,
               beta=1.,
               l1=0.1,
               l2=1.0,
               bits=20)

    loss = 0.
    count = 0
    for t, line in enumerate(DictReader(open(train), delimiter='\t')):
        clf.fit(line)
        print "Line ", line, "\n"
        pred = clf.predict()
        print "Prediction ", pred, "\n"
        loss += clf.logloss_new()
        print "LogLoss ", loss, "\n"
        clf.update(pred)
        count += 1
        break
        if count % 10000 == 0:
            print ("(seen, loss) : ", (count, loss * 1. / count))
        if count == 100000:
            break


    exit()


    test = 'data/testSearchStream.tsv'
    with open('temp.csv', 'w') as output:
        for t, line in enumerate(DictReader(open(test), delimiter='\t')):
            clf.fit(line)
            output.write('%s\n' % str(clf.predict()))

    sample = pd.read_csv('data/sampleSubmission.csv')
    preds = np.array(pd.read_csv('temp.csv', header=None))
    index = sample.ID.values - 1

    sample['IsClick'] = preds[index]
    sample.to_csv('submission.csv', index=False)

