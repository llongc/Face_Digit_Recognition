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

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """

    # might be useful in your code later...
    # this is a list of all features in the training set.

    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    print self.features
    print len(self.features)
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]

    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

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
    train_label = util.Counter()
    train_total = {}
    train_feature = {}
    for i in self.features:
        train_feature[i] = {}
        train_feature[i][0] = util.Counter()
        train_feature[i][1] = util.Counter()
        train_feature[i][2] = util.Counter()
        train_feature[i][3] = util.Counter()
        train_feature[i][4] = util.Counter()
        train_total[i] = util.Counter()
    # get the probability of P(y) and p(xi | y) from the features of training data and labels
    # train_label is for count of P(y)
    # train_feature is for count of P(xi | y)
    for i in range(len(trainingLabels)):
        # print trainingLabels[i]
        train_label[trainingLabels[i]] += 1
        for j in trainingData[i]:
            index = trainingData[i][j] // 10
            train_feature[j][index][train_label[trainingLabels[i]]] += 1
            train_total[j][train_label[trainingLabels[i]]] += 1

    # total count for labels
    self.total_count = train_label

    # select_model = util.Counter()
    maxProb = None
    maxAccur = 0
    for kg in kgrid:
        prob_feature = {}
        for i in self.features:
            prob_feature[i] = {}
            prob_feature[i][0] = util.Counter()
            prob_feature[i][1] = util.Counter()
            prob_feature[i][2] = util.Counter()
            prob_feature[i][3] = util.Counter()
            prob_feature[i][4] = util.Counter()
            for j in range(5):
                for k in self.legalLabels:
                    # this is smoothing part
                    # print "k is ", kg
                    prob_feature[i][j][k] = float((train_feature[i][j][k] + kg)) / (train_total[i][k] + 3 * kg)
                    if prob_feature[i][j][k] == 0.0:
                        print "Warning!!!!!!!!!!!!!!!!!!"
                        print i, j, k
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
    print "finish learning"

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
    # print self.prob

    for label in self.legalLabels:
        logJoint[label] += math.log( float(self.total_count[label]) / self.total_count.totalCount())
        for fi in datum:
            # print self.prob[fi]
            fea = datum[fi] // 10
            # print fi, fea, label
            # print self.prob[fi][fea][label]
            logJoint[label] += math.log(self.prob[fi][fea][label])

    #     for feature in self.features:
    #         print self.prob[feature][label]
    #         logJoint[label] += math.log(self.prob[feature][label])
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
