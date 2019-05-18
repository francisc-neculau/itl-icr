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

# from os import environ, chdir
#
# from keras import models, layers, optimizers
# from keras.preprocessing.image import ImageDataGenerator
#
#
# environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# root = "C:\ComputerScience\\resources\[Processed] InftyCDB-3_new\\"
#
# # Setting Image and Data Generators
# train_idg = ImageDataGenerator()
# train_g = train_idg.flow_from_directory(directory=root + "train",
#                                         target_size=(32, 32),
#                                         color_mode="grayscale",
#                                         class_mode='categorical',
#                                         batch_size=5000,
#                                         shuffle=True)
#
# valid_idg = ImageDataGenerator()
# valid_g = valid_idg.flow_from_directory(directory=root + "test",
#                                         target_size=(32, 32),
#                                         color_mode="grayscale",
#                                         class_mode='categorical',
#                                         batch_size=1000,
#                                         shuffle=True)
#
# # CNN Architecture
# my_model = models.Sequential()
# my_model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),
#                            use_bias=True, input_shape=(32, 32, 1)))
# my_model.add(layers.Activation('relu'))
#
# my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# my_model.add(layers.Dropout(rate=0.1))
#
# my_model.add(layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), use_bias=True))
# my_model.add(layers.Activation('relu'))
#
# my_model.add(layers.Conv2D(filters=36, kernel_size=(3, 3), strides=(1, 1), use_bias=True))
# my_model.add(layers.Activation('relu'))
#
# my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
# my_model.add(layers.Flatten())
#
# my_model.add(layers.Dense(units=512, use_bias=True))
# my_model.add(layers.Activation('relu'))
#
# my_model.add(layers.Dense(units=10, use_bias=True))
# my_model.add(layers.Activation('softmax'))
#
# # CNN Summary
# print(my_model.summary())
#
# # Model Loss function and Optimizer method
# compile = my_model.compile(optimizer=optimizers.sgd(lr=0.1, decay=0.01), #rmsprop(lr=0.1, decay=0.01)
#                            loss='categorical_crossentropy',
#                            metrics=['accuracy'])
#
# # Training Options
# fit = my_model.fit_generator(generator=train_g,
#                              steps_per_epoch=22,
#                              epochs=100,
#                              verbose=1,
#                              # callbacks=callb_l,
#                              validation_data=valid_g,
#                              validation_steps=1)
#
# # Saving Model
# my_model.save(filepath=root + 'cnn.h5', overwrite=True)



