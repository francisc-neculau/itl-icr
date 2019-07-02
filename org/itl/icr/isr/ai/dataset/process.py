import cv2 as cv
import numpy as np
import random


class DataAugmentation:

    width = 40;
    height = 40;

    def augment(self, code_to_char_images, label_to_codes):
        # for each family of codes
        for codes in label_to_codes.values():
            number_of_char_images = 0;
            # for each code
            for code in codes:
                number_of_char_images = number_of_char_images + len(code_to_char_images[code])
            if number_of_char_images > 5000:
                extra = number_of_char_images - 5000
                for code in codes:
                    random.shuffle(code_to_char_images[code])
                    for i in range (0, int(extra / len(codes))):
                        code_to_char_images[code].pop()
            else:
                missing = 5000 - number_of_char_images
                for code in codes:
                    char_images = self.__augment_images(code_to_char_images[code], int(missing / len(codes)))
                    code_to_char_images[code].extend(char_images)
        return code_to_char_images

    @staticmethod
    def __augment_images(char_images, number_to_add):
        new_char_images = []
        for i in range(0, number_to_add):
            new_char_images.append(np.copy(char_images[random.randint(0, len(char_images) - 1)]))
        return new_char_images


class DataProcessor:
    """
    """

    @staticmethod
    def square_resize(charImage):
        height = charImage.shape[0]
        width = charImage.shape[1]

        new_size = (DataAugmentation.width, DataAugmentation.height)

        if height > width:
            left = int((height - width)/2)
            right = height - width - left
            result = cv.copyMakeBorder(charImage, 0, 0, left, right, cv.BORDER_CONSTANT, None, value=255)
            result = cv.resize(result, new_size)
        elif height < width:
            top = int((width - height) / 2)
            bottom = width - height - top
            result = cv.copyMakeBorder(charImage, top, bottom, 0, 0, cv.BORDER_CONSTANT, None, value=255)
            result = cv.resize(result, new_size)
        else:
            result = cv.resize(charImage, new_size)
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
