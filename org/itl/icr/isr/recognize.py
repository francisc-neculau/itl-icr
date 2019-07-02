from builtins import staticmethod
from org.itl.icr.isr.ai.nn.cnn2 import CnnModel
from org.itl.icr.isr.ai.dataset.infty_cdb3 import char_type_registry, CharType
from PIL import Image as PillowImage
import cv2 as cv
import numpy as np
import logging


class ImageSymbolClassifier:
    def __init__(self):
        self.cnnModel = CnnModel()
        self.cnnModel.load()

    @staticmethod
    def __is_anything_between(first_char_image, second_char_image, y_axis_to_char_images):
        """
            This method check to see if there is any other
            char_image between the 2 provided.

                         +----------+
                         |          |
        (y_min, x_min) <-A----------+

                              +----------B-> (y_max, x_max)
                              |          |
                              +----------+
        :return: True/False
        """
        # FIXME: This may be done with polygon intersections.
        br1 = first_char_image.bounding_rectangle
        br2 = second_char_image.bounding_rectangle
        y_min = min(br1.bottom_right_point[1], br2.bottom_right_point[1]) + 1
        x_min = min(br1.top_left_point[0], br2.top_left_point[0])
        y_max = max(br1.top_left_point[1], br2.top_left_point[1]) - 1
        x_max = max(br1.bottom_right_point[0], br2.bottom_right_point[0])

        for y in range(y_min, y_max):
            if y not in y_axis_to_char_images:
                continue
            char_images = y_axis_to_char_images[y]
            for char_image in char_images:
                ci_x = char_image.center.mass_x_y[0]
                ci_y = char_image.center.mass_x_y[1]
                if x_min <= ci_x <= x_max and y_min <= ci_y <= y_max:
                    return True
        return False

    @staticmethod
    def __is_above(char_image, above_char_image):
        return char_image.bounding_rectangle.is_complete_above(
            above_char_image.bounding_rectangle
        )

    @staticmethod
    def __recognize_equal_char_images(minus_char_images, y_axis_to_char_images, image):
        equal_char_images = []
        disposable = []
        disposable_char_images = []
        minuses_length = len(minus_char_images)
        for i in range(0, minuses_length):
            if i in disposable:
                continue
            for j in range(0, minuses_length):
                if j in disposable or i == j:
                    continue
                # distance is not greater than half of width
                distance = minus_char_images[i].center.distance_from(
                    minus_char_images[j].center
                )
                if distance > minus_char_images[i].width / 2:
                    continue
                # nothing between the 2 minus signs
                if ImageSymbolClassifier.__is_anything_between(
                    minus_char_images[i], minus_char_images[j], y_axis_to_char_images
                ):
                    continue
                # FIXME: Check that bounding boxes align ?
                # Two equal signs should have bounding boxes
                # that align on their x Axis.

                # FIXME: Maybe a double check with the AT ?
                # Here we are assuming that we found a minus but
                # maybe it would be best to double check with
                # the AI-Recognizer and 'rollback' in case of
                # contradiction.
                equals_char_image = minus_char_images[i].merge_in(minus_char_images[j], image)
                equals_char_image.char_type = char_type_registry.find_by_external_name(CharType.EXTERNAL_NAME_equals)
                equal_char_images.append(equals_char_image)
                disposable.append(i)
                disposable.append(j)
                disposable_char_images.append(minus_char_images[i])
                disposable_char_images.append(minus_char_images[j])
                break
        return equal_char_images, disposable_char_images

    @staticmethod
    def __recognize_fraction_char_images(minus_char_images, y_axis_to_char_images):
        fraction_char_images = []
        disposable_char_images = []
        for minus_char_image in minus_char_images:
            fraction_numerator = None
            for y in y_axis_to_char_images.keys():
                char_images = y_axis_to_char_images[y]
                for char_image in char_images:
                    # comparisons to self are not valid !
                    if char_image.identifier == minus_char_image.identifier:
                        continue
                    # do we have a possible numerator ?
                    is_above = ImageSymbolClassifier.__is_above(minus_char_image, char_image)
                    if not is_above:
                        continue
                    # is it reasonable close to the minus ?
                    char_image_y = char_image.bounding_rectangle.bottom_right_point[1]
                    if char_image_y < minus_char_image.bounding_rectangle.top_left_point[1] - minus_char_image.width * 0.6:
                        continue
                    # is nothing between them ?
                    is_anything_between = ImageSymbolClassifier.__is_anything_between(
                        minus_char_image, char_image, y_axis_to_char_images
                    )
                    if is_anything_between:
                        continue
                    # we found a numerator!
                    fraction_numerator = char_image

                if fraction_numerator is not None:
                    break
            if fraction_numerator is not None:
                fraction_char_image = minus_char_image
                fraction_char_image.char_type = char_type_registry.get_special_char_type(CharType.IDENTIFIER_FRACTION)
                fraction_char_images.append(fraction_char_image)
                disposable_char_images.append(minus_char_image)
        return fraction_char_images, disposable_char_images

    def __recognize_ij_char_images(self, dot_char_images, y_axis_to_char_images, image):
        ij_char_images = []
        disposable_char_images = []
        for dot_char_image in dot_char_images:
            mass_x_y = dot_char_image.center.mass_x_y
            bottom_right_point = dot_char_image.bounding_rectangle.bottom_right_point

            # FIXME: Are these good number ? Use constants ?
            # These values I have decided for have no
            # formal background. Are the result of
            # trial and error.
            x_min = int(mass_x_y[0] - 0.1 * mass_x_y[0])
            x_max = int(mass_x_y[0] + 0.1 * mass_x_y[0])
            y_min = bottom_right_point[1] + 1
            y_max = y_min + dot_char_image.height * 4

            # We need only to find the first withing
            # the above boundaries if such a char_image exists
            match = None
            for y in range(y_min, y_max):
                if y not in y_axis_to_char_images:
                    continue
                char_images = y_axis_to_char_images[y]
                for char_image in char_images:
                    cm_x = char_image.center.mass_x_y[0]
                    br_tl_y = char_image.bounding_rectangle.top_left_point[1]
                    if x_min <= cm_x <= x_max and y_min <= br_tl_y <= y_max:
                        match = char_image
                        break
                if match is not None:
                    break
            if match is not None:
                ij_char_image = dot_char_image.merge_in(match, image)
                # FIXME: Maybe not the best solution!
                ij_char_image.char_type = self.cnnModel.predict_raw_image_char_type(ij_char_image.image)
                ij_char_images.append(ij_char_image)
                disposable_char_images.append(match)
                disposable_char_images.append(dot_char_image)
        return ij_char_images, disposable_char_images

    @staticmethod
    def __external_name_to_char_images(char_images):
        external_name_to_char_images = {}
        for char_image in char_images:
            char_type = char_image.char_type
            if char_type.external_name not in external_name_to_char_images:
                external_name_to_char_images[char_type.external_name] = []
            external_name_to_char_images[char_type.external_name].append(char_image)
        return external_name_to_char_images

    def classify(self, image, image_symbols):
        char_images = []
        y_axis_to_char_images = {}
        for char_image in image_symbols:
            char_image.char_type = self.cnnModel.predict_raw_image_char_type(char_image.image)
            y = char_image.bounding_rectangle.top_left_point[1]
            if y not in y_axis_to_char_images:
                y_axis_to_char_images.update({y: [char_image]})
            else:
                y_axis_to_char_images[y].append(char_image)

            char_images.append(char_image)

        if logging.getLogger("root").isEnabledFor(logging.DEBUG):
            drawn_image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            drawn_image = CharImageUtil.draw_char_images_on_image(drawn_image, char_images)
            PillowImage.fromarray(drawn_image, 'RGB').show(title="Coarse Predictions")

        # Case: Square Root sign
        for char_image in char_images:
            for other_char_image in char_images:
                # skip self check.
                if char_image.identifier == other_char_image.identifier:
                    continue
                if char_image.contains(other_char_image):
                    char_image.char_type = char_type_registry.get_special_char_type(CharType.IDENTIFIER_SQUARE_ROOT)
                    break

        # Case: =
        #   When we have two minuses that form an equal
        external_name_to_char_images = ImageSymbolClassifier.__external_name_to_char_images(char_images)
        if CharType.EXTERNAL_NAME_minus in external_name_to_char_images:
            minus_char_images = external_name_to_char_images[CharType.EXTERNAL_NAME_minus]
            equal_char_images, disposable_char_images = ImageSymbolClassifier.__recognize_equal_char_images(
                minus_char_images, y_axis_to_char_images, image
            )
            disposable_identifiers = [char_image.identifier for char_image in disposable_char_images]
            char_images = [char_image for char_image in char_images
                           if char_image.identifier not in disposable_identifiers
                           ]
            char_images.extend(equal_char_images)

        # Case: - | Escaped Fractions
        #   When we have a minus that has a numerator ( and a denominator ).
        external_name_to_char_images = ImageSymbolClassifier.__external_name_to_char_images(char_images)
        if CharType.EXTERNAL_NAME_minus in external_name_to_char_images:
            minus_char_images = external_name_to_char_images[CharType.EXTERNAL_NAME_minus]
            fraction_char_images, disposable_char_images = ImageSymbolClassifier.__recognize_fraction_char_images(
                minus_char_images, y_axis_to_char_images
            )
            disposable_identifiers = [char_image.identifier for char_image in disposable_char_images]
            char_images = [char_image for char_image in char_images
                           if char_image.identifier not in disposable_identifiers
                           ]
            char_images.extend(fraction_char_images)

        # Case: i, j
        #   No need to recompute the variable external_name_to_char_images
        if CharType.EXTERNAL_NAME_dot in external_name_to_char_images:
            dot_char_images = external_name_to_char_images[CharType.EXTERNAL_NAME_dot]
            ij_char_images, disposable_char_images = self.__recognize_ij_char_images(
                dot_char_images, y_axis_to_char_images, image
            )
            disposable_identifiers = [char_image.identifier for char_image in disposable_char_images]
            char_images = [char_image for char_image in char_images
                           if char_image.identifier not in disposable_identifiers
                           ]
            char_images.extend(ij_char_images)

        if logging.getLogger("root").isEnabledFor(logging.DEBUG):
            drawn_image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            drawn_image = CharImageUtil.draw_char_images_on_image(drawn_image, char_images)
            PillowImage.fromarray(drawn_image, 'RGB').show(title="Good Predictions")

        return char_images


class CharImageUtil:
    @staticmethod
    def draw_char_images_on_image(image, char_images):
        drawn_image = np.copy(image)
        for char_image in char_images:
            caption = char_image.identifier + \
                      " " + \
                      char_image.char_type.external_name
            # caption = char_image.char_type.internal_name
            drawn_image = char_image.draw_on_image(drawn_image, caption)
        return drawn_image

    @staticmethod
    def order_by_y_axis(char_images):
        y_axis_to_char_images = {}
        for char_image in char_images:
            y = char_image.bounding_rectangle.top_left_point[1]
            if y not in y_axis_to_char_images:
                y_axis_to_char_images.update({y: [char_image]})
            else:
                y_axis_to_char_images[y].append(char_image)
        return y_axis_to_char_images

    @staticmethod
    def order_by_x_axis(char_images):
        x_axis_to_char_images = {}
        for char_image in char_images:
            x = char_image.bounding_rectangle.top_left_point[0]
            if x not in x_axis_to_char_images:
                x_axis_to_char_images.update({x: [char_image]})
            else:
                x_axis_to_char_images[x].append(char_image)
        return x_axis_to_char_images

    @staticmethod
    def order_by_external_name(char_images):
        external_name_to_char_images = {}
        for char_image in char_images:
            char_type = char_image.char_type
            if char_type.external_name not in external_name_to_char_images:
                external_name_to_char_images[char_type.external_name] = []
            external_name_to_char_images[char_type.external_name].append(char_image)
        return external_name_to_char_images
