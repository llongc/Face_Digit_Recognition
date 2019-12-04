# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import dataClassifier
import timeit

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 2 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    start_time = timeit.default_timer()
    """
    Outside shell to call your method. Do not modify this method.
    """

    # might be useful in your code later...
    # this is a list of all features in the training set.

    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    # print self.features
    # print len(self.features)
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]

    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
    elapsed = timeit.default_timer() - start_time
    return elapsed

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):

    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter
    that gives the best accuracy on the held-out validationData.

    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"

    # initialize the counter
    # count how many for each label 0 - 9
    train_label = util.Counter()
    # count the sum of possibilities for feature i based on result y
    train_total = {}
    # count the possibilities for each feature i for each class based on result y
    train_feature = {}
    for i in self.features:
        train_feature[i] = {}
        for a in range(dataClassifier.CLASS):
            train_feature[i][a] = util.Counter()
        train_total[i] = util.Counter()
    # get the probability of P(y) and p(xi | y) from the features of training data and labels
    # train_label is for count of P(y)
    # train_feature is for count of P(xi | y)
    for i in range(len(trainingLabels)):
        # print trainingLabels[i]
        train_label[trainingLabels[i]] += 1
        for j in trainingData[i]:
            # print trainingLabels[i]
            # print j, trainingData[i][j], train_label[trainingLabels[i]]
            # print trainingLabels[i]
            train_feature[j][trainingData[i][j]][trainingLabels[i]] += 1
            train_total[j][trainingLabels[i]] += 1
        # break

    # total count for labels
    self.total_count = train_label

    # select_model = util.Counter()
    maxProb = None
    maxAccur = 0
    for kg in kgrid:
        prob_feature = {}
        for i in self.features:
            prob_feature[i] = {}
            for a in range(dataClassifier.CLASS):
                prob_feature[i][a] = util.Counter()
            for j in range(dataClassifier.CLASS):
                for k in self.legalLabels:
                    # this is smoothing part
                    # print "k is ", kg
                    prob_feature[i][j][k] = float((train_feature[i][j][k] + kg)) / (train_total[i][k] + dataClassifier.CLASS * kg)
        self.prob = prob_feature
        guess = self.classify(validationData)
        accurate = 0
        for i in range(len(validationLabels)):
            if validationLabels[i] == guess[i]:
                accurate += 1
        if (float(accurate) / len(validationLabels)) > maxAccur:
            maxAccur = float(accurate) / len(validationLabels)
            maxProb = self.prob
            self.k = kg
    self.prob = maxProb
    # print "finish learning"

    # util.raiseNotDefined()





  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.

    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses

  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """
    logJoint = util.Counter()

    for label in self.legalLabels:
        logJoint[label] += math.log( float(self.total_count[label]) / self.total_count.totalCount())
        for fi in datum:
            # print fi, datum[fi], label, self.prob[fi], self.prob[fi][datum[fi]]
            if fi not in self.prob:
                print "@@@@@"
                print fi, datum[fi]
            logJoint[label] += math.log(self.prob[fi][datum[fi]][label])
    # "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    return logJoint

  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2)

    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
