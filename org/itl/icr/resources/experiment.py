from org.itl.icr.iss.segment import CharacterSegmentation, Util
from org.itl.icr.isr.recognize import ImageSymbolClassifier
from org.itl.icr.isr.ai.dataset.paths import Paths
from PIL import Image as PillowImage
import logging

import cv2 as cv



# chars = "1234567890"
#
# # for char in chars:
# #     print(char + " " + str(ord(char)))
#
# for _c in chars: print(_c + " " + ('U+%04x' % ord(_c)))
#
#
# exit(1)


















# logging.DEBUG
# logging.INFO
# logging.WARNING
# logging.ERROR
# logging.CRITICAL
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)-8s %(message)s')
logger = logging.getLogger("root")
logger.setLevel(logging.ERROR)
# logger.setLevel(logging.DEBUG)


# FIXME: Bad Segmentation!
filePath = Paths.equations() + "clean\\10.jpg"
# filePath = Paths.equations() + "clean\\6.jpg"
# filePath = Paths.equations() + "clean\\5.jpg"
# filePath = Paths.equations() + "clean\\0.jpg"
filePath = Paths.equations() + "clean\\12.jpg"
# filePath = Paths.equations() + "clean\\8.jpg" # BAD : letter L
filePath = Paths.character_palette()
# # filePath = Paths.equations() + "raw\\handwritten.jpg" # BAD : Equals
# filePath = Paths.equations() + "clean\\7.jpg"
# filePath = Paths.equations() + "clean\\13.jpg" # BAD : Greater Thank or Equals
# filePath = Paths.equations() + "clean\\2.jpg"
# filePath = Paths.equations() + "clean\\15.jpg"
# filePath = Paths.equations() + "clean\\11.jpg"

# filePath = Paths.equations() + "clean\\0.jpg"
image = cv.imread(filePath, cv.IMREAD_GRAYSCALE)

characterSegmentation = CharacterSegmentation()
char_images = characterSegmentation.segment(image)


result = Util.draw_char_images(cv.cvtColor(image, cv.COLOR_GRAY2RGB), char_images)
PillowImage.fromarray(result).show(title="Unclassified")


# serialize(char_images, Paths.resources() + "result.txt");

imageSymbolClassifier = ImageSymbolClassifier()
classified_char_images = imageSymbolClassifier.classify(image, char_images)

result = Util.draw_char_images(cv.cvtColor(image, cv.COLOR_GRAY2RGB), classified_char_images)
PillowImage.fromarray(result).show(title="Result")

cv.imshow('Result', result)
cv.waitKey(0)
