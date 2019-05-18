import logging
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
#   @see: https://stackoverflow.com/questions/26955017/matplotlib-crashing-python
#   @see: https://stackoverflow.com/questions/44281342/changing-matplotlib-backend-crashes-debug-mode


class DataAnalytics:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze(self, codeToCharImages, codeToTranslation):
        translationToCodes = {}
        for code in codeToTranslation.keys():
            translation = codeToTranslation[code];
            if code not in codeToCharImages.keys():
                continue
            if translation not in translationToCodes:
                translationToCodes[translation] = [code]
            else:
                translationToCodes[translation].append(code)
        values = []  # in same order as traversing keys
        keys = []  # also needed to preserve order
        for translation in translationToCodes.keys():
            keys.append(translation.split('-')[1])
            values.append(sum((len(codeToCharImages[code]) for code in translationToCodes[translation])))
        plt.bar(keys, values, color='g')
        plt.show()

    def __shapeAnalysis(self, codeToCharImages, codeToTranslation):

        self.logger.info("All Char classes : ")
        self.logger.info(len(codeToTranslation.keys()))
        self.logger.info(codeToTranslation)
        self.logger.info("")

        self.logger.info("Char classes aggregated : ")
        self.logger.info(len(codeToCharImages.keys()))
        self.logger.info(codeToCharImages.keys())
        self.logger.info("")

        self.logger.info("Char classes not found in the data set : ")
        self.logger.info([codeToTranslation[k] for k in codeToTranslation if k in codeToCharImages])
        self.logger.info("")
        self.logger.info("")

        self.logger.info("Informations about extracted chars :")
        for key in codeToCharImages.keys():

            maxHeight = 0
            sumHeight = 0
            minHeight = 1000

            maxWidth = 0
            sumWidth = 0
            minWidth = 1000

            maxHwRatio = 0
            hwRatioSum = 0
            minHwRatio = 1000

            for charImage in codeToCharImages[key]:
                height = charImage.shape[0]
                width = charImage.shape[1]
                hwRatio = height/width

                hwRatioSum += hwRatio
                minHwRatio = min(minHwRatio, hwRatio)
                maxHwRatio = max(maxHwRatio, hwRatio)

                minHeight = min(minHeight, height)
                maxHeight = max(maxHeight, height)
                sumHeight += height

                minWidth = min(minWidth, width)
                maxWidth = max(maxWidth, width)
                sumWidth += width

            self.logger.info("-category : " + codeToTranslation[key])
            self.logger.info("-size : " + str(len(codeToCharImages[key])))
            self.logger.info("-hw ration min/max/avg: " + "{0:.2f}".format(minHwRatio) + "/" + "{0:.2f}".format(maxHwRatio) + "/"
                             + "{0:.2f}".format(hwRatioSum/len(codeToCharImages[key])))
            self.logger.info("-height min/max/avg: " + str(minHeight) + "/" + str(maxHeight) + "/"
                             + str(sumHeight/len(codeToCharImages[key])))
            self.logger.info("-width min/max/avg: " + str(minWidth) + "/" + str(maxWidth) + "/"
                             + str(sumWidth/len(codeToCharImages[key])))
            self.logger.info("")
