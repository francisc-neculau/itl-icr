import cv2 as cv
import numpy as np


class DataAugmentation:

    def augment(self, codeToCharImages, codeToTranslation):
        translationToCodes = {}
        for code in codeToTranslation.keys():
            translation = codeToTranslation[code];
            if code not in codeToCharImages.keys():
                continue
            if translation not in translationToCodes:
                translationToCodes[translation] = [code]
            else:
                translationToCodes[translation].append(code)

        for codes in translationToCodes.values():
            for code in codes:
                augmentedImage = self.__augmentImages(codeToCharImages[code], int(5000/len(codes)))
                codeToCharImages[code].extend(augmentedImage)

        return codeToCharImages

    def __augmentImages(self, images, max=5000):

        newImages = []
        count = max - len(images)
        if count <= 0:
            return []
        else:
            index = 0;
            while count > 0:
                count -= 1
                newImages.append(np.copy(images[index % len(images)]))
                index += 1

        return newImages


class DataProcessor:
    """
    """

    @staticmethod
    def squareResize(charImage):
        height = charImage.shape[0]
        width = charImage.shape[1]

        newSize = (32, 32)

        if height > width:
            left = int((height - width)/2)
            right = height - width - left
            result = cv.copyMakeBorder(charImage, 0, 0, left, right, cv.BORDER_CONSTANT, None, value=255)
            result = cv.resize(result, newSize)
        elif height < width:
            top = int((width - height) / 2)
            bottom = width - height - top
            result = cv.copyMakeBorder(charImage, top, bottom, 0, 0, cv.BORDER_CONSTANT, None, value=255)
            result = cv.resize(result, newSize)
        else:
            result = cv.resize(charImage, newSize)
        return result

    @staticmethod
    def augment(images, max=5000):
        newImages = []
        count = max - len(images)
        if count <= 0:
            return []
        else:
            index = 0;
            while count > 0:
                count -= 1
                newImages.append(np.copy(images[index % len(images)]))
                index += 1

        return newImages

    @staticmethod
    def __resize(charImage):
        height = charImage.shape[0]
        width = charImage.shape[1]

        newWidth = 37#52;
        ratio = height/width
        newHeight = int(ratio * newWidth)

        return cv.resize(charImage, (newWidth, newHeight))

    @staticmethod
    def __pad(charImage):
        height = charImage.shape[0]
        width = charImage.shape[1]
        maxHeight = 188 + 12
        maxWidth = 120 + 10
        top = int((maxHeight - height)/2) # shape[0] = rows
        bottom = top
        top = top - ((2 * top + height) - maxHeight) # adjust for +-1

        left = int((maxWidth - width)/2) # shape[1] = cols
        right = left
        left = left - ((2 * left + width) - maxWidth)  # adjust for +-1

        return cv.copyMakeBorder(charImage, top, bottom, left, right, cv.BORDER_CONSTANT, None, value=255)

# imagePath = "C:\\ComputerScience\\resources\\0.jpg"
# image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
#
# cv.imshow('Before', image)
# cv.waitKey(0)
# result = DataProcessor.augment(image)
# cv.imshow('After', result)
# cv.waitKey(0)