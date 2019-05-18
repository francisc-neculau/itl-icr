from org.itl.icr.api.protobuf import segmentation_result_pb2


def serialize(char_images, path):
    """
    Transforms a CharImage into it's protobuf representation
    and serializes this to the given paths.
    :param char_images: Char images to be serialized
    :param path: Path where to write the serialized char images
    :return: nothing
    """
    segmentation_result = segmentation_result_pb2.SegmentationResult()
    for char_image in char_images:
        serializable_char_image = segmentation_result.charImages.add()
        serializable_char_image.identifier = char_image.identifier
        serializable_char_image.charTypeName = char_image.char_type.external_name
        serializable_char_image.height = char_image.height
        serializable_char_image.width = char_image.width
        serializable_char_image.centerOfMass.x = char_image.center.mass_x_y[0]
        serializable_char_image.centerOfMass.y = char_image.center.mass_x_y[1]
        serializable_char_image.boundingRectangle.topLeft.x = char_image.bounding_rectangle.top_left_point[0]
        serializable_char_image.boundingRectangle.topLeft.y = char_image.bounding_rectangle.top_left_point[1]
        serializable_char_image.boundingRectangle.bottomRight.x = char_image.bounding_rectangle.bottom_right_point[0]
        serializable_char_image.boundingRectangle.bottomRight.y = char_image.bounding_rectangle.bottom_right_point[1]
    f = open(path, "wb")
    f.write(segmentation_result.SerializeToString())
    f.close()
