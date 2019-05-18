import sys,os

sys.path.append(os.getcwd())
sys.path.append('C:\\ComputerScience\\source\\itl-icr')


from org.itl.icr.iss.segment import CharacterSegmentation
from org.itl.icr.isr.ai.dataset.InftyCDB3B import CharTypeRegistry
from org.itl.icr.isr.recognize import ImageSymbolClassifier
from org.itl.icr.isr.ai.dataset.paths import Paths
from org.itl.icr.api.protobuf.util import serialize
from pathlib import Path
import logging
import cv2 as cv


# One of : DEBUG INFO WARNING ERROR CRITICAL
logger_level = logging.ERROR
# logging.basicConfig(level=logging.ERROR, format='%(asctime)-15s %(module) %(levelname)-8s %(message)s')
logging.getLogger("root").setLevel(logger_level)
logging.getLogger(CharacterSegmentation.__name__).setLevel(logger_level)

if len(sys.argv) != 3:
    raise ValueError("Missing mandatory path argument.")

image_file_path = Path(sys.argv[1])
if not image_file_path.is_file():
    raise ValueError("Path argument must be a valid image path.")

result_file_path = Path(sys.argv[2])

image = cv.imread(str(image_file_path), cv.IMREAD_GRAYSCALE)

characterSegmentation = CharacterSegmentation()
char_images = characterSegmentation.segment(image)

imageSymbolClassifier = ImageSymbolClassifier()
classified_char_images = imageSymbolClassifier.classify(image, char_images)

serialize(classified_char_images, str(result_file_path));
