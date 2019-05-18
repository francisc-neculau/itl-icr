import csv
import random
import math
import operator

class Knn:

    def __init__(self, trainingDataPoints):
        """

        :param trainingDataPoints: list of shape [[[features], [class]]]
        """
        self.trainingDataPoints = trainingDataPoints

    def euclideanDistance(self, dataPointFeatures, trainingDataPointFeatures):
        distance = 0
        for i in range(len(trainingDataPointFeatures)):
            distance += pow((dataPointFeatures[i] - trainingDataPointFeatures[i]), 2)
        return math.sqrt(distance)

    def classify(self, dataPointFeatures, k=1):

        classDistancePairs = []
        for i in range(len(self.trainingDataPoints)):
            distance = self.euclideanDistance(dataPointFeatures,
                                              self.trainingDataPoints[i][0])
            classDistancePairs.append((self.trainingDataPoints[i][1],
                                       distance))
        classDistancePairs.sort(key=operator.itemgetter(1))

        neighbors = []
        for i in range(k):
            neighbors.append(classDistancePairs[i][0])

        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

        return sortedVotes[0][0]

        # self.__moments = None
        # self.__huInvariants = None
        # self.__features = None
    # @property
    # def moments(self):
    #     if self.__moments is None:
    #         self.__moments = cv.moments(cv.bitwise_not(self.image), binaryImage=True)
    #     return self.__moments
    #
    # @property
    # def huMoments(self):
    #     if self.__huInvariants is None:
    #         self.__huInvariants = cv.HuMoments(self.moments)
    #     return self.__huInvariants
    #
    # @property
    # def features(self):
    #     if self.__features is None:
    #         self.__features = [[huMoment[0] for huMoment in self.huMoments], [self.name]]
    #     return self.__features