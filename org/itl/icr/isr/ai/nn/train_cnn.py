from org.itl.icr.isr.ai.nn.cnn2 import CnnModel

cnnWrapper = CnnModel()
cnnWrapper.train().save()

# cnnWrapper.load()
#
# path = "C:\\ComputerScience\\resources\\[Processed] InftyCDB-3_new\\train\\Numeric-five-0x4135-0x0135\\0.jpg"
# image = cv.imread(path, cv.IMREAD_GRAYSCALE)
#
# label = cnnWrapper.predictImageClass(image)
#
# cv.imshow(label, image)
# cv.waitKey(0)
# print(label)
#
