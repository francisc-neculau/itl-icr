from shapely.geometry import Polygon
import cv2 as cv
import numpy as np
from math import sqrt

X_AXIS_COLLISION_PRECISION = 4
Y_AXIS_COLLISION_PRECISION = 4


class Colors:
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    pink = (255, 0, 144)
    white = (255, 255, 255)
    turquoise = (175, 238, 238)


class Center:
    def __init__(self, min_enclosing_circle_x_y, mass_x_y):
        self.min_enclosing_circle_x_y = min_enclosing_circle_x_y
        self.mass_x_y = mass_x_y

    def distance_from(self, center):
        return sqrt(
            (self.mass_x_y[0] - center.mass_x_y[0]) ** 2 +
            (self.mass_x_y[1] - center.mass_x_y[1]) ** 2
        )

    def is_between_x(self, x_min, x_max):
        return x_min <= self.mass_x_y[0] <= x_max

    def is_between_y(self, y_min, y_max):
        return y_min <= self.mass_x_y[1] <= y_max

    @staticmethod
    def merge(c1, c2):
        # FIXME: I'm not sure that this is correct!
        min_enclosing_circle_x_y = (
            int((c1.min_enclosing_circle_x_y[0] + c2.min_enclosing_circle_x_y[0]) / 2),
            int((c1.min_enclosing_circle_x_y[1] + c2.min_enclosing_circle_x_y[1]) / 2)
        )
        mass_x_y = (
            int((c1.mass_x_y[0] + c2.mass_x_y[0]) / 2),
            int((c1.mass_x_y[1] + c2.mass_x_y[1]) / 2)
        )
        return Center(min_enclosing_circle_x_y, mass_x_y)


class BoundingRectangle:
    def __init__(self, open_cv_bounding_rectangle):
        """
            Constructs a BoundingRectangle object.
        :param open_cv_bounding_rectangle: It must be a list of the form
            [x, y, x_offset, y_offset]
        """
        self.top_left_point = (
            open_cv_bounding_rectangle[0],
            open_cv_bounding_rectangle[1]
        )
        self.x_offset = open_cv_bounding_rectangle[2]
        self.y_offset = open_cv_bounding_rectangle[3]
        self.bottom_right_point = (
            self.top_left_point[0] + self.x_offset,
            self.top_left_point[1] + self.y_offset
        )

    def is_complete_above(self, above_br):
        """
        Checks weather above_br is above self. It will
        return true only in the case :
              +-----+
              |  B  |
              +-----+
            +---------+
            |    A    |
            +---------+
        """
        if self.top_left_point[0] <= above_br.top_left_point[0] and \
           self.bottom_right_point[0] >= above_br.bottom_right_point[0] and \
           self.top_left_point[1] <= above_br.bottom_right_point[1]:
            return True
        return False

    @staticmethod
    def merge(br1, br2):
        x = min(br1.top_left_point[0], br2.top_left_point[0])
        y = min(br1.top_left_point[1], br2.top_left_point[1])
        x_offset = max(br1.x_offset, br2.x_offset)
        y_offset = max(br1.bottom_right_point[1], br2.bottom_right_point[1]) - y
        return BoundingRectangle([x, y, x_offset, y_offset])

    def crop_image(self, image):
        return image[
               self.top_left_point[1]:self.bottom_right_point[1],
               self.top_left_point[0]:self.bottom_right_point[0]
            ]


class CharImage:
    """
        (x, y)----------+
            |           |
            |           |
            |           |
            +-----------(x+w, y+h)
    """

    def __init__(self, identifier, image, center, contours, bounding_rectangle):
        self.identifier = identifier
        # FIXME: Maybe refactor this name to smth like raw_image
        self.char_type = None
        self.image = image
        # FIXME: Should accept only binary image ?
        self.height, self.width = self.image.shape

        self.center = center
        # list of convex hulls !
        # FIXME: Document the shape of this thing!
        self.contours = contours
        self.bounding_rectangle = bounding_rectangle
        # FIXME: Maybe a list o polygons also ??
        self.__polygon = None

    @property
    def polygon(self):
        if self.__polygon is None:
            # FIXME: Here we should merge all convex hulls ?
            convex_hull_points = [
                (self.contours[0][k][0][0], self.contours[0][k][0][1])
                for k in range(0, len(self.contours[0]))
            ]
            self.__polygon = Polygon(convex_hull_points)
        return self.__polygon

    def merge_in(self, image_symbol, image):
        """
            Merges self with the image_symbol.
        :return: merged symbol image
        """
        bounding_rectangle = BoundingRectangle.merge(
            self.bounding_rectangle,
            image_symbol.bounding_rectangle
        )
        crop_image = bounding_rectangle.crop_image(image)
        identifier = self.identifier + "-" + image_symbol.identifier
        contours = []
        contours.extend(self.contours)
        contours.extend(image_symbol.contours)
        center = Center.merge(self.center, image_symbol.center)
        return CharImage(
            identifier,
            crop_image,
            center,
            contours,
            bounding_rectangle
        )

    def centroids_collide(self, image_symbol):
        c1 = self.center.min_enclosing_circle_x_y
        c2 = image_symbol.center.min_enclosing_circle_x_y
        result = abs(c1[0] - c2[0]) < X_AXIS_COLLISION_PRECISION
        result &= abs(c1[1] - c2[1]) < Y_AXIS_COLLISION_PRECISION
        return result

    def is_part_of(self, other_char_image):
        """
            Checks if self is a part of imageSymbol.
        :param other_char_image:
        :return: true/false
        """
        return other_char_image.contains(self) & other_char_image.covers(self)

    def covers(self, image_symbol):
        """
            Checks if self's polygon contour fully covers
        the imageSymbol's polygon contour.
        :param image_symbol:
        :return: true/false
        """
        return self.polygon.covers(image_symbol.polygon)

    def contains(self, other_char_image):
        """
            Checks if self's bounding box contains the
        other_char_image's bounding box.
        :param other_char_image:
        :return: true/false
        """
        self_bb = self.bounding_rectangle
        other_bb = other_char_image.bounding_rectangle
        result = True
        result &= self_bb.top_left_point[0] <= other_bb.top_left_point[0]
        result &= self_bb.top_left_point[1] <= other_bb.top_left_point[1]
        # opposite vertex now..
        result &= self_bb.bottom_right_point[0] >= other_bb.bottom_right_point[0]
        result &= self_bb.bottom_right_point[1] >= other_bb.bottom_right_point[1]
        return result

    def draw_on_image(self, image, caption=None):
        if caption is None:
            if self.char_type is not None:
                caption = self.identifier + " " + self.char_type.external_name
            else:
                caption = "id:" + self.identifier
        drawn_image = np.copy(image)
        for contour in self.contours:
            cv.drawContours(drawn_image, [contour], 0, Colors.green)
        tlp = (
            self.bounding_rectangle.top_left_point[0] - 1,
            self.bounding_rectangle.top_left_point[1] - 1
        )
        brp = (
            self.bounding_rectangle.bottom_right_point[0] + 1,
            self.bounding_rectangle.bottom_right_point[1] + 1
        )
        cv.rectangle(
            drawn_image,
            tlp,
            brp,
            Colors.blue,
            thickness=1
        )
        cv.circle(drawn_image, self.center.min_enclosing_circle_x_y, 2, Colors.pink, -1)
        cv.circle(drawn_image, self.center.mass_x_y, 2, Colors.turquoise, -1)
        cv.putText(
            img=drawn_image,
            text=caption,
            org=(
                self.bounding_rectangle.top_left_point[0],
                self.bounding_rectangle.top_left_point[1]-2
            ),
            fontFace=cv.QT_FONT_NORMAL,
            fontScale=0.4,
            color=Colors.red,
            thickness=1,
            lineType=cv.LINE_AA
        )
        return drawn_image

    def show(self):
        cv.imshow('Symbol', self.image)
        cv.waitKey(0)

    def equals(self, other):
        return self.identifier == other.identifier

    def to_string(self):
        string = ''
        string += 'shape:' + str(self.image.shape)
        string += ' '
        string += 'id:' + str(self.identifier)
        return string
