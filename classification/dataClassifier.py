# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness
# code for data classification

import Knear
import naiveBayes
import perceptron
import mira
import samples
import sys
import util
import numpy as np
import timeit
# from sklearn.metrics import accuracy_score


TEST_SET_SIZE = 100 # It is indeed the percent for test dataset, default is 10%

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
DIVIDE = 1
CLASSDIV = 1
# print CLASSDIV
CLASS = (DIVIDE * DIVIDE / CLASSDIV if DIVIDE * DIVIDE % CLASSDIV == 0 else DIVIDE * DIVIDE / CLASSDIV + 1) + 1
# print CLASS

def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.

  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).

  ## DESCRIBE YOUR ENHANCED FEATURES HERE...

  ##
  """
  # features =  basicFeatureExtractorDigit(datum)
  features = util.Counter()
  xd = DIGIT_DATUM_WIDTH // DIVIDE
  if DIGIT_DATUM_WIDTH % DIVIDE != 0:
      xd += 1
  yd = DIGIT_DATUM_HEIGHT // DIVIDE
  if DIGIT_DATUM_HEIGHT % DIVIDE != 0:
      yd += 1

  for i in range(xd):
      for j in range(yd):
          features['area'+str(i)+str(j)] = 0
  for x in range(DIGIT_DATUM_WIDTH):
      for y in range(DIGIT_DATUM_HEIGHT):
          if datum.getPixel(x, y) > 0:
              # features['allblack'] += 1
              if CLASS == 2:
                  features['area'+str(x // DIVIDE)+str(y // DIVIDE)] = 1
              else:
                  features['area'+str(x // DIVIDE)+str(y // DIVIDE)] += 1

  for f in features:
      if features[f] != 0:
          tmp = features[f] // CLASSDIV if features[f] % CLASSDIV == 0 else features[f] // CLASSDIV + 1
          # if tmp >= 2:
              # print f, features[f], CLASSDIV
          features[f] = tmp

  # print features
  return features


def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features = util.Counter()
  xd = FACE_DATUM_WIDTH // DIVIDE
  if FACE_DATUM_WIDTH % DIVIDE != 0:
      xd += 1
  yd = FACE_DATUM_HEIGHT // DIVIDE
  if FACE_DATUM_HEIGHT % DIVIDE != 0:
      yd += 1
  # print xd, yd
  for i in range(xd):
      for j in range(yd):
          features['area'+str(i)+str(j)] = 0
  for x in range(FACE_DATUM_WIDTH):
      for y in range(FACE_DATUM_HEIGHT):
          if datum.getPixel(x, y) > 0:
              # features['allblack'] += 1
              if CLASS == 2:
                  features['area'+str(x // DIVIDE)+str(y // DIVIDE)] = 1
              else:
                  features['area'+str(x // DIVIDE)+str(y // DIVIDE)] += 1

  for f in features:
      if features[f] != 0:
          tmp = features[f] // CLASSDIV if features[f] % CLASSDIV == 0 else features[f] // CLASSDIV + 1
          if tmp >= 2:
              print f, features[f], CLASSDIV
          features[f] = tmp

  # print features
  return features

def analysis(time, acc, std):

  data_percent = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  for i in range(len(data_percent)):
      print "For testing data size " + str(data_percent[i]) + "%\tTraining Time: " +\
        str(time[i]) + " sec\tAccuracy " + str(acc[i] * 100) + "%\tStderr "+str(std[i])


  # # Put any code here...
  # # Example of use:
  # for i in range(len(guesses)):
  #     prediction = guesses[i]
  #     truth = testLabels[i]
  #     # print testData[0]
  #     # print rawTestData[0]
  #     if (prediction != truth):
  #         print "==================================="
  #         print "Mistake on example %d" % i
  #         print "Predicted %d; truth is %d" % (prediction, truth)
  #         print "Image: "
  #         print rawTestData[i]
  #         # printImage(rawTestData[i].getPixels())
  #         # print rawTestData[i][0]
  #         # printImage(rawTestData[i])
  #         break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.

      Pixels should take the form
      [(2,2), (2, 3), ...]
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print "new features:", pix
            continue
      print image

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser
  parser = OptionParser(USAGE_STRING)

  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['knear', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='knear')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}

  # Set up variables according to the command line input.
  print "Doing classification"
  print "--------------------"
  print "data:\t\t" + options.data
  print "classifier:\t\t" + options.classifier
  if not options.classifier == 'minicontest':
    print "using enhanced features?:\t" + str(options.features)
  else:
    print "using minicontest feature extractor"
  print "training set size:\t" + str(options.training)+"%"
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
    if (options.classifier == 'minicontest'):
      featureFunction = contestFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace
  else:
    print "Unknown dataset", options.data
    print USAGE_STRING
    sys.exit(2)

  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)

  if options.training <= 0:
    print "Training set size should be a positive integer (you provided: %d)" % options.training
    print USAGE_STRING
    sys.exit(2)

  if options.smoothing <= 0:
    print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
    print USAGE_STRING
    sys.exit(2)

  if options.odds:
    if options.label1 not in legalLabels or options.label2 not in legalLabels:
      print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
      print USAGE_STRING
      sys.exit(2)

  if(options.classifier == "knear"):
    classifier = Knear.KnearestNeighbourClassifier(legalLabels)
  elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print "using automatic tuning for naivebayes"
        classifier.automaticTuning = True
    else:
        print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  elif(options.classifier == "mira"):
    classifier = mira.MiraClassifier(legalLabels, options.iterations)
    if (options.autotune):
        print "using automatic tuning for MIRA"
        classifier.automaticTuning = True
    else:
        print "using default C=0.001 for MIRA"
  elif(options.classifier == 'minicontest'):
    import minicontest
    classifier = minicontest.contestClassifier(legalLabels)
  else:
    print "Unknown classifier:", options.classifier
    print USAGE_STRING

    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage

  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']

  # Load data
  numTraining = options.training
  numTest = options.test

  if(options.data=="faces"):
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
  else:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


  # Extract features
  print "Extracting features..."
  trainingData = map(featureFunction, rawTrainingData)
  validationData = map(featureFunction, rawValidationData)
  testData = map(featureFunction, rawTestData)

  # Conduct training and testing
  print "Training..."
  elapse = classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  print elapse
  print "Validating..."
  guesses = classifier.classify(validationData)
  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  # print guesses
  # print validationLabels
  print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
  print "Testing..."
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
  # analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

  # do odds ratio computation if specified at command line
  if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
    label1, label2 = options.label1, options.label2
    features_odds = classifier.findHighOddsFeatures(label1,label2)
    if(options.classifier == "naiveBayes" or options.classifier == "nb"):
      string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
    else:
      string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

    print string3
    printImage(features_odds)

  if((options.weights) & (options.classifier == "perceptron")):
    for l in classifier.legalLabels:
      features_weights = classifier.findHighWeightFeatures(l)
      print ("=== Features with high weight for label %d ==="%l)
      printImage(features_weights)

def selfRunClassifier():
    print "Doing classification"
    print "--------------------"

    data_percent = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print "=================Digits===================="


    # NaiveBayes part
    # print "Training by using NaiveBayes Algorithm"
    # featureFunction = enhancedFeatureExtractorDigit
    # legalLabels = range(10)
    # classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    # lst_avg_time = []
    # lst_avg_acc = []
    # lst_std_acc = []
    # for percent in data_percent:
    #     print "training set size:\t" + str(percent)+"%"
    #     # print "setSmoothing: k value is ", classifier.k
    #     lst_time = []
    #     lst_acc = []
    #     for i in range(5):
    #         start_time = timeit.default_timer()
    #         rawTrainingData = samples.loadDataFile("digitdata/trainingimages", percent,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    #         trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", percent)
    #         rawValidationData = samples.loadDataFile("digitdata/validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    #         validationLabels = samples.loadLabelsFile("digitdata/validationlabels", TEST_SET_SIZE)
    #         rawTestData = samples.loadDataFile("digitdata/testimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    #         testLabels = samples.loadLabelsFile("digitdata/testlabels", TEST_SET_SIZE)
    #         # print "Extracting features..."
    #         trainingData = map(featureFunction, rawTrainingData)
    #         validationData = map(featureFunction, rawValidationData)
    #         testData = map(featureFunction, rawTestData)
    #         classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    #         elapse = timeit.default_timer() - start_time
    #         lst_time.append(elapse)
    #         guesses = classifier.classify(testData)
    #         correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    #         lst_acc.append(float(correct) / len(testLabels))
    #     print '--------------------------------------------------------------'
    #     lst_avg_time.append(np.mean(lst_time))
    #     lst_avg_acc.append(np.mean(lst_acc))
    #     lst_std_acc.append(np.std(lst_acc))
    # analysis(lst_avg_time, lst_avg_acc, lst_std_acc)

    # Perceptron algorithm
    # print "Training by using Perceptron Algorithm"
    print "Training by using NaiveBayes Algorithm"
    featureFunction = enhancedFeatureExtractorDigit
    legalLabels = range(10)
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    lst_avg_time = []
    lst_avg_acc = []
    lst_std_acc = []
    for percent in data_percent:
        print "training set size:\t" + str(percent)+"%"
        # print "setSmoothing: k value is ", classifier.k
        lst_time = []
        lst_acc = []
        for i in range(5):
            start_time = timeit.default_timer()
            rawTrainingData = samples.loadDataFile("digitdata/trainingimages", percent,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
            trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", percent)
            rawValidationData = samples.loadDataFile("digitdata/validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
            validationLabels = samples.loadLabelsFile("digitdata/validationlabels", TEST_SET_SIZE)
            rawTestData = samples.loadDataFile("digitdata/testimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
            testLabels = samples.loadLabelsFile("digitdata/testlabels", TEST_SET_SIZE)
            # print "Extracting features..."
            trainingData = map(featureFunction, rawTrainingData)
            validationData = map(featureFunction, rawValidationData)
            testData = map(featureFunction, rawTestData)
            classifier.train(trainingData, trainingLabels, validationData, validationLabels)
            elapse = timeit.default_timer() - start_time
            lst_time.append(elapse)
            guesses = classifier.classify(testData)
            correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
            lst_acc.append(float(correct) / len(testLabels))
        print '--------------------------------------------------------------'
        lst_avg_time.append(np.mean(lst_time))
        lst_avg_acc.append(np.mean(lst_acc))
        lst_std_acc.append(np.std(lst_acc))
    analysis(lst_avg_time, lst_avg_acc, lst_std_acc)
    # Percentron algorithm






    # K nearest neighbour algorithm
    print "Training by using KNN Algorithm"
    print "Only use 10% test set"
    # featureFunction = enhancedFeatureExtractorDigit
    # legalLabels = range(10)
    # classifier = perceptron.PerceptronClassifier(legalLabels, 3)
    # lst_avg_time = []
    # lst_avg_acc = []
    # lst_std_acc = []
    # for percent in data_percent:
    #     print "training set size:\t" + str(percent)+"%"
    #     # print "setSmoothing: k value is ", classifier.k
    #     lst_time = []
    #     lst_acc = []
    #     for i in range(5):
    #         start_time = timeit.default_timer()
    #         rawTrainingData = samples.loadDataFile("digitdata/trainingimages", percent,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    #         trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", percent)
    #         rawValidationData = samples.loadDataFile("digitdata/validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    #         validationLabels = samples.loadLabelsFile("digitdata/validationlabels", TEST_SET_SIZE)
    #         rawTestData = samples.loadDataFile("digitdata/testimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    #         testLabels = samples.loadLabelsFile("digitdata/testlabels", TEST_SET_SIZE)
    #         # print "Extracting features..."
    #         trainingData = map(featureFunction, rawTrainingData)
    #         validationData = map(featureFunction, rawValidationData)
    #         testData = map(featureFunction, rawTestData)
    #
    #         classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    #         elapse = timeit.default_timer() - start_time
    #         # print elapse
    #         lst_time.append(elapse)
    #         guesses = classifier.classify(testData)
    #         correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    #         lst_acc.append(float(correct) / len(testLabels))
    #     print '--------------------------------------------------------------'
    #     lst_avg_time.append(np.mean(lst_time))
    #     lst_avg_acc.append(np.mean(lst_acc))
    #     lst_std_acc.append(np.std(lst_acc))
    # analysis(lst_avg_time, lst_avg_acc, lst_std_acc)

    # K nearest neighbour algorithm
    # print "Training by using KNN Algorithm"
    # print "Only use 10% test set"
    #
    # lst_avg_time = []
    # lst_avg_acc = []
    # lst_std_acc = []
    # for percent in data_percent:
    #     print "training set size:\t" + str(percent)+"%"
    #     lst_time = []
    #     lst_acc = []
    #     for i in range(5):
    #         start_time = timeit.default_timer()
    #         featureFunction = enhancedFeatureExtractorDigit
    #         legalLabels = range(10)
    #         classifier = Knear.KnearestNeighbourClassifier(legalLabels)
    #         rawTrainingData = samples.loadDataFile("digitdata/trainingimages", percent,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    #         trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", percent)
    #         rawValidationData = samples.loadDataFile("digitdata/validationimages", 10,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    #         validationLabels = samples.loadLabelsFile("digitdata/validationlabels", 10)
    #         rawTestData = samples.loadDataFile("digitdata/testimages", 10,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    #         testLabels = samples.loadLabelsFile("digitdata/testlabels", 10)
    #         # print "Extracting features..."
    #         trainingData = map(featureFunction, rawTrainingData)
    #         validationData = map(featureFunction, rawValidationData)
    #         testData = map(featureFunction, rawTestData)
    #
    #         classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    #         elapse = timeit.default_timer() - start_time
    #         # print elapse
    #         lst_time.append(elapse)
    #         guesses = classifier.classify(testData)
    #         correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    #         lst_acc.append(float(correct) / len(testLabels))
    #     print '--------------------------------------------------------------'
    #     lst_avg_time.append(np.mean(lst_time))
    #     lst_avg_acc.append(np.mean(lst_acc))
    #     lst_std_acc.append(np.std(lst_acc))
    # analysis(lst_avg_time, lst_avg_acc, lst_std_acc)






    print ""
    print "=================Faces===================="
    # NaiveBayes Algorithm
    # print "Training by using NaiveBayes Algorithm"
    # # featureFunction = enhancedFeatureExtractorDigit
    # # legalLabels = range(2)
    # # classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    # lst_avg_time = []
    # lst_avg_acc = []
    # lst_std_acc = []
    # for percent in data_percent:
    #     print "training set size:\t" + str(percent)+"%"
    #     lst_time = []
    #     lst_acc = []
    #     for i in range(5):
    #         featureFunction = enhancedFeatureExtractorDigit
    #         legalLabels = range(2)
    #         classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    #         start_time = timeit.default_timer()
    #         rawTrainingData = samples.loadDataFile("facedata/facedatatrain", percent,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    #         trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", percent)
    #         rawValidationData = samples.loadDataFile("facedata/facedatatrain", TEST_SET_SIZE,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    #         validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", TEST_SET_SIZE)
    #         rawTestData = samples.loadDataFile("facedata/facedatatest", TEST_SET_SIZE,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    #         testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", TEST_SET_SIZE)
    #         trainingData = map(featureFunction, rawTrainingData)
    #         validationData = map(featureFunction, rawValidationData)
    #         testData = map(featureFunction, rawTestData)
    #         classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    #         elapse = timeit.default_timer() - start_time
    #         lst_time.append(elapse)
    #         guesses = classifier.classify(testData)
    #         correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    #         lst_acc.append(float(correct) / len(testLabels))
    #     print '--------------------------------------------------------------'
    #     lst_avg_time.append(np.mean(lst_time))
    #     lst_avg_acc.append(np.mean(lst_acc))
    #     lst_std_acc.append(np.std(lst_acc))
    # analysis(lst_avg_time, lst_avg_acc, lst_std_acc)

    # Perceptron Algorithm
    print "Training by using Perceptron Algorithm"
    # featureFunction = enhancedFeatureExtractorDigit
    # legalLabels = range(2)
    # classifier = perceptron.PerceptronClassifier(legalLabels, 3)
    lst_avg_time = []
    lst_avg_acc = []
    lst_std_acc = []
    for percent in data_percent:
        print "training set size:\t" + str(percent)+"%"
        lst_time = []
        lst_acc = []
        for i in range(5):
            featureFunction = enhancedFeatureExtractorDigit
            legalLabels = range(2)
            classifier = perceptron.PerceptronClassifier(legalLabels, 3)
            start_time = timeit.default_timer()
            rawTrainingData = samples.loadDataFile("facedata/facedatatrain", percent,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
            trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", percent)
            rawValidationData = samples.loadDataFile("facedata/facedatatrain", TEST_SET_SIZE,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
            validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", TEST_SET_SIZE)
            rawTestData = samples.loadDataFile("facedata/facedatatest", TEST_SET_SIZE,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
            testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", TEST_SET_SIZE)
            trainingData = map(featureFunction, rawTrainingData)
            validationData = map(featureFunction, rawValidationData)
            testData = map(featureFunction, rawTestData)
            elapse = classifier.train(trainingData, trainingLabels, validationData, validationLabels)
            lst_time.append(elapse)
            guesses = classifier.classify(testData)
            correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
            lst_acc.append(float(correct) / len(testLabels))
        print '--------------------------------------------------------------'
        lst_avg_time.append(np.mean(lst_time))
        lst_avg_acc.append(np.mean(lst_acc))
        lst_std_acc.append(np.std(lst_acc))
    analysis(lst_avg_time, lst_avg_acc, lst_std_acc)





    # # Knn Algorithm
    # print "Training by using KNN Algorithm"
    # print "Only use 10% test set"
    # # featureFunction = enhancedFeatureExtractorDigit
    # # legalLabels = range(2)
    # # classifier = Knear.KnearestNeighbourClassifier(legalLabels)
    # lst_avg_time = []
    # lst_avg_acc = []
    # lst_std_acc = []
    # for percent in data_percent:
    #     print "training set size:\t" + str(percent)+"%"
    #     lst_time = []
    #     lst_acc = []
    #     for i in range(5):
    #         start_time = timeit.default_timer()
    #         featureFunction = enhancedFeatureExtractorDigit
    #         legalLabels = range(2)
    #         classifier = Knear.KnearestNeighbourClassifier(legalLabels)
    #         rawTrainingData = samples.loadDataFile("facedata/facedatatrain", percent,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    #         trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", percent)
    #         rawValidationData = samples.loadDataFile("facedata/facedatatrain", 10,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    #         validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", 10)
    #         rawTestData = samples.loadDataFile("facedata/facedatatest", 10,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    #         testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", 10)
    #         trainingData = map(featureFunction, rawTrainingData)
    #         validationData = map(featureFunction, rawValidationData)
    #         testData = map(featureFunction, rawTestData)
    #         classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    #         elapse = timeit.default_timer() - start_time
    #         lst_time.append(elapse)
    #         guesses = classifier.classify(testData)
    #         correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    #         lst_acc.append(float(correct) / len(testLabels))
    #     print '--------------------------------------------------------------'
    #     lst_avg_time.append(np.mean(lst_time))
    #     lst_avg_acc.append(np.mean(lst_acc))
    #     lst_std_acc.append(np.std(lst_acc))
    # analysis(lst_avg_time, lst_avg_acc, lst_std_acc)


if __name__ == '__main__':
  # Read input
  # args, options = readCommand( sys.argv[1:] )
  # Run classifier
  # runClassifier(args, options)
  selfRunClassifier()
