from org.itl.icr.iss.model.symbol import CharImage, BoundingRectangle, Center
from PIL import Image as PillowImage
import cv2 as cv
import numpy as np
import logging
from org.itl.icr.isr.ai.dataset.paths import Paths


class CharacterSegmentation:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def __binarize_image(self, greyscale_image):
        self.logger.info("Binarizing the image with min-threshold:100 and max-threshold:255")
        _, binary_image = cv.threshold(greyscale_image, 100, 255, cv.THRESH_BINARY)
        if logging.getLogger("root").isEnabledFor(logging.DEBUG):
            PillowImage.fromarray(binary_image).show(title="Binary Image")
        return binary_image

    def __invert_image(self, image):
        self.logger.info("Inverting image")
        inverted_image = cv.bitwise_not(image)
        if logging.getLogger("root").isEnabledFor(logging.DEBUG):
            PillowImage.fromarray(inverted_image).show(title="Inverted Image")
        return inverted_image

    def __compute_image_edges_map(self, inverted_image):
        self.logger.info("Computing the image edges map with canny edge detection algorithm")
        result = cv.Canny(inverted_image, threshold1=1, threshold2=255, apertureSize=5, L2gradient=True)
        if logging.getLogger("root").isEnabledFor(logging.DEBUG):
            PillowImage.fromarray(result).show(title="Edges Map")

        result = cv.morphologyEx(result, op=cv.MORPH_CLOSE, kernel=np.ones((2, 2), np.uint8))
        if logging.getLogger("root").isEnabledFor(logging.DEBUG):
            PillowImage.fromarray(result).show(title="Closed-Edges Map")
        return result

    def __extract_char_images(self, extraction_image, mask_image):
        _, contours, hierarchy = cv.findContours(extraction_image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        self.logger.info("Extracting dirty char images from binary image")
        char_images = []
        for i, contour in enumerate(contours):
            convex_hull = cv.convexHull(contour)
            if len(convex_hull) < 3:
                # This cannot be a valid symbol. More likely noise.
                continue
            bounding_rectangle = BoundingRectangle(cv.boundingRect(contour))
            c, _ = cv.minEnclosingCircle(contour)
            min_enclosing_circle_x_y = (int(c[0]), int(c[1]))
            # FIXME: This may be slow !
            M = cv.moments(contour)
            mass_x_y = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            center = Center(min_enclosing_circle_x_y, mass_x_y)

            char_image = CharImage(
                str(i),
                # FIXME: Crop but with a mask ? so that we have clean radicals for example
                bounding_rectangle.crop_image(mask_image),
                center,
                [contour],
                bounding_rectangle
            )
            char_images.append(char_image)

        self.logger.info(str(len(char_images)) + " dirty char images found")
        if logging.getLogger("root").isEnabledFor(logging.DEBUG):
            image = Util.draw_char_images(cv.cvtColor(mask_image, cv.COLOR_GRAY2RGB), char_images)
            PillowImage.fromarray(image, 'RGB').show(title="Dirty Segments")

        return char_images

    def __clean_char_images(self, dirty_char_images, binary_image):
        self.logger.info("Removing dirty char images")
        number_of_symbols = len(dirty_char_images)
        disposable = []
        for i in range(0, number_of_symbols):
            if i in disposable:
                continue
            for j in range(0, number_of_symbols):
                if j in disposable or i == j:
                    continue
                if dirty_char_images[i].centroids_collide(dirty_char_images[j]):
                    disposable.append(j)
                    continue

                if dirty_char_images[j].is_part_of(dirty_char_images[i]):
                    disposable.append(j)

        clean_char_images = [dirty_char_images[i] for i in range(0, number_of_symbols) if i not in disposable]
        self.logger.info(str(len(disposable)) + " dirty char images removed")
        if logging.getLogger("root").isEnabledFor(logging.DEBUG):
            image = Util.draw_char_images(cv.cvtColor(binary_image, cv.COLOR_GRAY2RGB), clean_char_images)
            PillowImage.fromarray(image, 'RGB').show(title="Clean Segments")

        return clean_char_images

    def segment(self, greyscale_image):
        """
            Segments a greyscale image of a Mathematical equation.
        The image must be greyscale with the equation written with
        black on a white background.
        :param greyscale_image:
        :return:
        """
        if not greyscale_image.any():
            raise ValueError('greyscaleImage parameter is empty')
        if len(greyscale_image.shape) != 2:
            raise ValueError('greyscaleImage is not grayscale')

        if logging.getLogger("root").isEnabledFor(logging.DEBUG):
            PillowImage.fromarray(greyscale_image).show(title="Greyscale Image")

        binary_image = self.__binarize_image(greyscale_image)
        inverted_binary_image = self.__invert_image(binary_image)
        # image_edges_map = self.__compute_image_edges_map(inverted_binary_image)
        # dirty_char_images = self.__extract_char_images(image_edges_map, binary_image)
        dirty_char_images = self.__extract_char_images(inverted_binary_image, binary_image)
        clean_char_images = self.__clean_char_images(dirty_char_images, binary_image)

        return clean_char_images

    def test(self):
        image = cv.imread(Paths.equations() + "clean\\10.jpg", cv.IMREAD_GRAYSCALE)
        binary_image = self.__binarize_image(image)
        inverted_binary_image = self.__invert_image(binary_image)
        char_images = self.__extract_char_images(inverted_binary_image, binary_image)

        result = Util.draw_char_images(cv.cvtColor(binary_image, cv.COLOR_GRAY2RGB), char_images)

        cv.imshow('Test', result)
        cv.waitKey(0)


class Util:
    @staticmethod
    def draw_char_images(drawing_board_image, char_images):
        drawn_image = np.copy(drawing_board_image)
        for char_image in char_images:
            drawn_image = char_image.draw_on_image(drawn_image)
        return drawn_image
