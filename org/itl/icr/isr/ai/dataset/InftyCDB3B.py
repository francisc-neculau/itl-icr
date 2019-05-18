from os import listdir
from os.path import isfile, join
import cv2 as cv
import os
import logging
import shutil
import time
import numpy as np
from sklearn.model_selection import train_test_split
from org.itl.icr.isr.ai.dataset.process import DataProcessor, DataAugmentation
from org.itl.icr.isr.ai.dataset.analysis import DataAnalytics
from org.itl.icr.isr.ai.dataset.paths import Paths

levels = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)-8s %(message)s')


class CharType:
    IDENTIFIER_FRACTION = "-103"
    IDENTIFIER_SQUARE_ROOT = "-104"

    EXTERNAL_NAME_i = "i"
    EXTERNAL_NAME_j = "j"
    EXTERNAL_NAME_1 = "1"
    EXTERNAL_NAME_dot = "."
    EXTERNAL_NAME_minus = "-"
    EXTERNAL_NAME_equals = "="
    EXTERNAL_NAME_fraction = "/"
    EXTERNAL_NAME_square_root = "sqrt"

    def __init__(self, identifier, codes, group, internal_name, external_name):
        self.identifier = identifier
        self.codes = codes
        self.group = group
        self.internal_name = internal_name
        self.external_name = external_name

    def is_numerical(self):
        return self.group == "Numeric"

    def is_letter(self):
        return self.is_roman_letter() | self.is_greek_letter()

    def is_roman_letter(self):
        return self.group == "Roman"

    def is_greek_letter(self):
        return self.group == "Greek"

    def is_parenthesis(self):
        return self.group == "Parenthesis"

    def is_fraction(self):
        return self.identifier == CharType.IDENTIFIER_FRACTION

    def equals(self, other):
        return self.identifier == other.identifier

    def create_label(self):
        return CharTypeRegistry.create_label(self.identifier, self.internal_name)


class CharTypeRegistry:
    __instance = None
    FRACTION_HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 0.05

    def __init__(self):
        self.ocrCodeListPath = Paths.dataset_inftycdb3() + "_custom\\Subset-Of-CDB3B-OcrCodeList.csv"
        self.label_to_char_type = {}
        self.code_to_label = {}
        self.external_name_to_char_type = {}

        # FIXME: Maybe this should go in some special files
        self.identifier_to_special_char_types = {}
        fraction = CharType("-103", "-0x1c2d", "BinaryOperator", "fraction", CharType.EXTERNAL_NAME_fraction)
        self.__add_special_char_type(fraction)
        square_root = CharType("-104", "-0x1c2e", "BinaryOperator", "square root", CharType.EXTERNAL_NAME_square_root)
        self.__add_special_char_type(square_root)

    def __add_special_char_type(self, char_type):
        self.identifier_to_special_char_types[char_type.identifier] = char_type
        label = char_type.create_label()
        self.label_to_char_type[label] = char_type
        self.code_to_label[char_type.codes] = label

    def get_special_char_type(self, char_type_identifier):
        return self.identifier_to_special_char_types[char_type_identifier]

    # FIXME: It would be nice to have it as a singleton!
    # def __new__(cls, val):
    #     if CharTypeRegistry.__instance is None:
    #         CharTypeRegistry.__instance = object.__new__(cls)
    #     CharTypeRegistry.__instance.val = val
    #     return CharTypeRegistry.__instance

    def load(self):
        file = open(self.ocrCodeListPath, 'r')
        # This file has the following header :
        # identifier,codes,group,internalName,externalName
        file.readline()  # skip header
        line = file.readline()
        while line:
            pieces = line.strip().split(',')
            identifier = pieces[0]
            codes = pieces[1].split('-')
            group = pieces[2]
            internal_name = pieces[3]
            external_name = pieces[4]

            char_type = CharType(identifier, codes, group, internal_name, external_name)
            label = CharTypeRegistry.create_label(identifier, internal_name)
            self.label_to_char_type[label] = char_type
            for code in codes:
                self.code_to_label[code] = label
            self.external_name_to_char_type[external_name] = char_type

            line = file.readline()

    def get_code_to_label(self):
        return self.code_to_label

    def find(self, label):
        return self.label_to_char_type[label]

    def find_by_external_name(self, external_name):
        return self.external_name_to_char_type[external_name]

    @staticmethod
    def create_label(identifier, internal_name):
        return identifier + "-" + internal_name


char_type_registry = CharTypeRegistry()
char_type_registry.load()


class Dataset:
    def __init__(self):
        self.readPath = Paths.dataset_inftycdb3()
        self.writePath = Paths.external_resources()

        self.reader = Reader(self.readPath, DataProcessor.squareResize)
        self.writer = Writer(self.writePath)

        self.code_to_char_images = None
        self.code_to_label = None
        self.label_to_codes = None

        self.dataAnalytics = DataAnalytics()
        self.dataAugmentation = DataAugmentation()

        self.data = None

    # FIXME: Refactor to number of char classes !
    def get_number_of_labels(self):
        return len(self.label_to_codes.keys())

    def load(self, display_data_analytics=False):
        """
            This method loads the InftyCDB3B datase.

        :param display_data_analytics: boolean value that indicates wether analitics
                                     about the data should be shown.
        :return: returns 2 numpy arrays. One representing the images
                 and another representing the labels.
        """
        self.code_to_char_images, self.code_to_label, self.label_to_codes = self.reader.read()
        if display_data_analytics:
            self.dataAnalytics.analyze(self.code_to_char_images, self.code_to_label)

        self.code_to_char_images = self.dataAugmentation.augment(self.code_to_char_images, self.code_to_label)
        if display_data_analytics:
            self.dataAnalytics.analyze(self.code_to_char_images, self.code_to_label)

        images = []
        labels = []
        for code in self.code_to_char_images.keys():
            for char_image in self.code_to_char_images[code]:
                images.append(char_image)
                labels.append(self.code_to_label[code])
        images = np.asarray(images)
        labels = np.asarray(labels)
        # randomize
        indices = np.arange(labels.shape[0])
        np.random.shuffle(indices)
        labels = labels[indices]
        images = images[indices]

        return images, labels

    def save(self):
        self.writer.write(self.code_to_char_images, self.code_to_label)


class Reader:
    """
        This class is used to read the InftyCDB-3-B Database.

        InftyCDB-3-B is an extract of InftyCDB-1, which includes data
    from 20 of its articles. To reduce the number of samples with
    the same character code, size, and shape, clustering was applied
    to the data from these 20 articles, reducing the number of data
    samples to about 70,000.
    The data is written in the same format as in InftyCDB-3-A.

    The folder structure should be like this :
        InftyCDB-3/InftyCDB-3_new (Folder)
                    |
                    |____InftyCDB-3-B (Folder)
                    |   |
                    |   |____images (Folder)
                    |   |
                    |   |____CharInfoDB-3-B_Info.txt | (B)
                    |
                    |____OcrCodeList.txt

    (B) is a csv file with the following header
    id, code, sheet, cx, cy, width, height

    FIXME: 0x4130 0x33f5 0x33f0 documenteaza-le !
    """

    def __init__(self, inftyCdb3FolderPath, resizeFunction=lambda x: x):
        """
            The parameter inftyCdb3FolderPath is the path to the
        main folder of the Database, InftyCDB-3/InftyCDB-3_new.
        """
        self.inftyCdb3FolderPath = inftyCdb3FolderPath
        self.char_code_file_path = inftyCdb3FolderPath + "_custom\\Subset-Of-CDB3B-OcrCodeList.csv"
        # self.char_code_file_path = inftyCdb3FolderPath + "_custom\\CDB3B-OcrCodeList.txt"
        self.char_info_file_path = inftyCdb3FolderPath + "\\InftyCDB-3-B\\CharInfoDB-3-B_Info.txt"
        self.images_folder_path = inftyCdb3FolderPath + "\\InftyCDB-3-B\\images"
        self.images_read_limit = 1
        self.resizeFunction = resizeFunction
        self.stepSize = 0.05
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_progress(self, finished_amount, total_amount, current_step):
        if finished_amount / total_amount > current_step:
            self.logger.info("{0:.2f}".format(current_step))
            return current_step + self.stepSize
        else:
            return current_step

    def __read_sheet_number_to_image(self):
        self.logger.info("Reading compressed char images: " + self.images_folder_path)

        sheet_number_to_image = {}
        image_file_names = listdir(self.images_folder_path)
        current_step = self.stepSize
        for i in range(0, len(image_file_names)):
            image_file_name = image_file_names[i]
            # imageName is also the sheet number from (B) but without the file extension
            image_file_path = join(self.images_folder_path, image_file_name)
            if isfile(image_file_path):
                image = cv.imread(image_file_path, cv.IMREAD_GRAYSCALE)
                sheet_number = image_file_name[:image_file_name.find('.')]
                sheet_number_to_image[sheet_number] = image
            else:
                self.logger.error("Not an image path : " + image_file_path)

            current_step = self.log_progress((i+1), len(image_file_names), current_step)
        return sheet_number_to_image

    def __read_code_to_char_images(self, sheet_number_to_image, codes):
        self.logger.info("Reading the char metadata file : " + self.char_info_file_path)
        char_info_file = open(self.char_info_file_path, 'r')
        char_info_file_size = os.path.getsize(self.char_info_file_path)

        line = char_info_file.readline()  # skip header.
        amount_read = len(line) + 1

        current_step = self.stepSize
        current_step = self.log_progress(amount_read, char_info_file_size, current_step)

        code_to_char_images = {}
        while True:
            line = char_info_file.readline()
            amount_read += len(line) + 1
            current_step = self.log_progress(amount_read, char_info_file_size, current_step)
            if not line:
                break

            pieces = line.split(',')
            code = '0x' + pieces[1]
            if code not in codes:
                continue
            sheet_number = pieces[2]
            x = int(pieces[3])
            y = int(pieces[4])
            width = int(pieces[5])
            height = int(pieces[6])

            # FIXME: This should be moved to the writter !
            char_image = self.resizeFunction(sheet_number_to_image[sheet_number][y:y+height, x:x+width])
            char_image = cv.bitwise_not(char_image)
            if code not in code_to_char_images:
                code_to_char_images[code] = []
            #  FIXME: Only for testing purposes
            # if len(code_to_char_images.get(code)) == self.images_read_limit:
            #     continue
            code_to_char_images.get(code).append(char_image)
        self.logger.info("Read the following codes : ")
        self.logger.info(code_to_char_images.keys())
        return code_to_char_images

    def __read_code_to_label(self):
        # FIXME: This will  be replaced with a call to CharTypeRegistry
        self.logger.info("Reading the char code file : " + self.char_code_file_path)

        char_code_csv_file = open(self.char_code_file_path, 'r')
        char_code_csv_file_size = os.path.getsize(self.char_code_file_path)
        char_code_csv_file.readline()  # skip header.
        line = char_code_csv_file.readline()
        amount_read = len(line) + 1

        current_step = self.stepSize;
        code_to_label = {}
        while line:
            pieces = line.strip().split(',')
            identifier = pieces[0]
            codes = pieces[1].split('-')
            internal_name = pieces[3]

            label = CharTypeRegistry.create_label(identifier, internal_name)
            for code in codes:
                code_to_label[code] = label

            if amount_read / char_code_csv_file_size > current_step:
                self.logger.info("{0:.2f}".format(current_step))
                current_step += self.stepSize
            line = char_code_csv_file.readline()
            amount_read += len(line) + 1
            current_step = self.log_progress(amount_read, char_code_csv_file_size, current_step)

        return code_to_label

    def read(self):
        sheet_number_to_image = self.__read_sheet_number_to_image()
        code_to_label = self.__read_code_to_label()
        code_to_char_images = self.__read_code_to_char_images(sheet_number_to_image, code_to_label.keys())

        label_to_codes = {}
        for code in code_to_label.keys():
            label = code_to_label[code]
            if label not in label_to_codes:
                label_to_codes[label] = []
                label_to_codes[label].append(code)

        return code_to_char_images, code_to_label, label_to_codes


class Writer:

    def __init__(self, write_folder_path):
        self.writeFolderPath = write_folder_path
        self.logger = logging.getLogger(self.__class__.__name__)

    def write(self, code_to_char_images, code_to_label):
        self.logger.info("Writing the char images : " + self.writeFolderPath)

        if os.path.exists(self.writeFolderPath):
            shutil.rmtree(self.writeFolderPath)
            time.sleep(4.0)
        os.makedirs(self.writeFolderPath)

        rootTrain = self.writeFolderPath + "\\train"
        rootTest = self.writeFolderPath + "\\test"
        os.makedirs(rootTrain)
        os.makedirs(rootTest)

        # rootTest =
        for charCode, codeTranslation in code_to_label.items():
            if charCode not in code_to_char_images:
                self.logger.warning("No char images found for " + charCode + "/" + codeTranslation)
                continue

            codeTranslationTrainFolder = rootTrain + "\\" + codeTranslation
            codeTranslationTestFolder = rootTest + "\\" + codeTranslation


            charImages = code_to_char_images[charCode]
            if not os.path.isdir(codeTranslationTrainFolder):
                os.makedirs(codeTranslationTrainFolder)
            if not os.path.isdir(codeTranslationTestFolder):
                os.makedirs(codeTranslationTestFolder)

            if len(charImages) > 10:
                trainCharImages, testCharImages = train_test_split(charImages, test_size = 0.1, random_state = 0)

                offset = len(os.listdir(codeTranslationTrainFolder))
                for i in range(0, len(trainCharImages)):
                    charImage = charImages[i]
                    cv.imwrite(codeTranslationTrainFolder + "\\" + str(offset + i) + ".jpg", charImage)

                offset = len(os.listdir(codeTranslationTestFolder))
                for i in range(0, len(testCharImages)):
                    charImage = charImages[i]
                    cv.imwrite(codeTranslationTestFolder + "\\" + str(offset + i) + ".jpg", charImage)
            else :
                # continue;
                # # FIXME: choose a different technique !
                offset = len(os.listdir(codeTranslationTestFolder))
                eroded = cv.erode(charImages[0], np.zeros((5, 5), np.uint8), iterations=1)
                cv.imwrite(codeTranslationTestFolder + "\\" + str(offset) + ".jpg", eroded)

                offset = len(os.listdir(codeTranslationTrainFolder))
                for i in range(0, len(charImages)):
                    charImage = charImages[i]
                    cv.imwrite(codeTranslationTrainFolder + "\\" + str(i) + ".jpg", charImage)
            self.logger.info("Written " + str(len(charImages)) + " of " + charCode + " | " + codeTranslation)


# dataset = Dataset()
# dataset.load()
# dataset.save()