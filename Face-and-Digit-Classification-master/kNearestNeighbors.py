# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
import numpy
import math
import statistics
import tracemalloc
import time
PRINT = True

class kNearestNeighborsClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, k=10):
    self.legalLabels = legalLabels
    self.type = "kNN"
    self.k = k
    self.weights = {}
  #   for label in legalLabels:
  #     self.weights[label] = util.Counter() # this is the data-structure you should use

  # def setWeights(self, weights):
  #   assert len(weights) == len(self.legalLabels);
  #   self.weights == weights;
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    self.trainingData = self.dataDownscale(trainingData)
    self.trainingLabels = trainingLabels
    


  def dataDownscale(self, datum_list):
    DATA_HEIGHT, DATA_WIDTH = 0,0
    BLOCK_HEIGHT, BLOCK_WIDTH = 0,0
    BLOCK_ROWS, BLOCK_COLS = 0,0
    if 2 in self.legalLabels:
      DATA_HEIGHT, DATA_WIDTH = 28,28
      BLOCK_HEIGHT, BLOCK_WIDTH = 4,4
      BLOCK_ROWS, BLOCK_COLS = 7,7

    else:
      DATA_HEIGHT, DATA_WIDTH = 70,60
      BLOCK_HEIGHT, BLOCK_WIDTH = 7,6
      BLOCK_ROWS, BLOCK_COLS = 10,10

    DDA = []
    for data in datum_list:
      dscaledData = util.Counter()
      for ib in range(BLOCK_ROWS):
        for jb in range(BLOCK_COLS):
          isFeature = 0

          for ism in range(BLOCK_HEIGHT):
            if isFeature:
              break
            for jsm in range(BLOCK_WIDTH):
              if data[( ib*BLOCK_HEIGHT + ism , jb*BLOCK_WIDTH + jsm )] == 1:
                isFeature = 1
                break

          dscaledData[(ib,jb)] = isFeature

      DDA.append(dscaledData)

    return DDA

  def distanceFind(self, test_datum, train_data):
    if True:
      x = test_datum - train_data
      return numpy.sum(numpy.abs([x[value] for value in x]))
    
  def classify(self, data ):
    """
    Find the k closest 'neighbors' of the test image in the training data
    and then return the label which appeared the most. If there is a tie
    then pick the label of the training image with the lowest distance.
    """

    data = self.dataDownscale(data)

    listGuess = []
    for datum in data:
      distanceList = []
      # print("------")
      # start = time.time()
      for i in range(len(self.trainingData)):
        distanceList.append(  (self.distanceFind(datum,self.trainingData[i]), i)  ) # need to pass i through for each distance to get the trainingLabel
      # end1 = time.time() - start
      # print(end1)

      distanceList.sort()
      distanceList = distanceList[:self.k]

      bestKlabels = []
      for distance in distanceList:
        bestKlabels.append(self.trainingLabels[distance[1]])

      try:
        listGuess.append(statistics.mode(bestKlabels))
      except:
        listGuess.append(bestKlabels[0])


    return listGuess


