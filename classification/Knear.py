# mostFrequent.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import numpy as np
# import dataClassifier
import timeit


class KnearestNeighbourClassifier(classificationMethod.ClassificationMethod):

  def __init__(self, legalLabels):
    self.k = 5
    self.guess = None
    self.type = "knear"

  def train(self, trainingData, trainingLabels, validationData, validationLabels):

    start_time = timeit.default_timer()
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    self.label = trainingLabels
    self.train = trainingData
    # self.length = len(trainingData[0])

    elapsed = timeit.default_timer() - start_time
    return elapsed


    # counter = util.Counter()
    # counter.incrementAll(labels, 1)
    # self.guess = counter.argMax()



  def classify(self, testData):
    """
    Classify all test data as the most common label.
    """
    # print self.guess
    # return [self.guess for i in testData]
    self.guess = np.zeros(len(testData),dtype = int)
    # print len(self.guess)
    # print len(testData)

    for i in range(len(testData)):
        kmost = util.Counter()
        count = util.Counter()
        for j in range(len(self.train)):
            dis = self.distance(testData[i], self.train[j])
            count[j] = dis
        # print count
        sorted = count.sortedKeys()
        # print sorted
        sorted.reverse()
        kmost.incrementAll([self.label[a] for a in sorted[:self.k]], 1)
        # print kmost.argMax()
        # print i
        self.guess[i] = kmost.argMax()
    return self.guess



  def distance(self, pic1, pic2):
      dis = 0
      for i in self.features:
          dis += (pic1[i] - pic2[i]) * (pic1[i] - pic2[i])
      dis = math.sqrt(dis)
      return dis
