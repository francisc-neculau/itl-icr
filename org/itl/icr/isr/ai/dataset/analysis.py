import logging
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
#   @see: https://stackoverflow.com/questions/26955017/matplotlib-crashing-python
#   @see: https://stackoverflow.com/questions/44281342/changing-matplotlib-backend-crashes-debug-mode


class DataAnalytics:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def analyze(code_to_char_images, code_to_translation):
        translation_to_codes = {}
        for code in code_to_translation.keys():
            translation = code_to_translation[code];
            if code not in code_to_char_images.keys():
                continue
            if translation not in translation_to_codes:
                translation_to_codes[translation] = [code]
            else:
                translation_to_codes[translation].append(code)
        values = []  # in same order as traversing keys
        keys = []  # also needed to preserve order
        for translation in translation_to_codes.keys():
            keys.append(translation.split('-')[1])
            values.append(sum((len(code_to_char_images[code]) for code in translation_to_codes[translation])))
        plt.bar(keys, values, color='g')
        plt.show()
