from keras import models, layers, optimizers
from keras.layers import MaxPooling2D, Conv2D, Dropout, Activation, Dense
from keras.models import model_from_json
from os import environ
from org.itl.icr.isr.ai.dataset.infty_cdb3 import Dataset, CharTypeRegistry, CharType, char_type_registry
from org.itl.icr.isr.ai.dataset.process import DataProcessor, DataAugmentation
from org.itl.icr.isr.ai.dataset.paths import Paths
import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelBinarizer
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CnnModel:
    def __init__(self):
        self.resourcesPath = Paths.nn_model()
        self.dataset = Dataset()
        self.model = None
        self.encoder = None
        self.trainPercentage = 0.8
        # self.validationPercentage = 0.2
        # self.testPercentage = ..

    def predict_image_char_type(self, image):
        return self.predict_images_char_type([image])[0]

    def predict_images_char_type(self, images):
        x = np.asarray(images)
        x = x.reshape((len(x), 32, 32, 1))
        x = x.astype('float32') / 255
        predictions = self.model.predict(x)
        labels = self.encoder.inverse_transform(predictions)
        return [char_type_registry.find(label) for label in labels]

    def predict_raw_images_char_type(self, raw_images):
        return [self.predict_raw_image_char_type(raw_image) for raw_image in raw_images]

    def predict_raw_image_char_type(self, binary_image):
        """
        This method will predict what char is represented in the
        given binary_image
        :param binary_image: image of a char
        :return: a char_type
        """
        height, width = binary_image.shape
        if height / width <= CharTypeRegistry.FRACTION_HEIGHT_TO_WIDTH_RATIO_THRESHOLD:
            char_type = char_type_registry.get_special_char_type(CharType.IDENTIFIER_FRACTION)
        else:
            image = cv.bitwise_not(DataProcessor.square_resize(binary_image))
            x = np.asarray(image)
            x = x.reshape((1, DataAugmentation.width, DataAugmentation.height, 1))
            x = x.astype('float32') / 255
            label = self.encoder.inverse_transform(self.model.predict(x))[0]
            char_type = char_type_registry.find(label)
        return char_type

    def load(self):
        """
            CNN Deserialization
        :return:
        """
        json_file = open(self.resourcesPath + "architecture.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.resourcesPath + "model_weights.h5")

        self.model = loaded_model

        self.encoder = LabelBinarizer()
        self.encoder.classes_ = np.load(self.resourcesPath + 'classes.npy')
        self.encoder.y_type_ = np.load(self.resourcesPath + 'y_type_.npy')
        self.encoder.sparse_input_ = np.load(self.resourcesPath + 'sparse_input_.npy')
        return self

    def save(self):
        """
            CNN Serialization
        :return:
        """
        with open(self.resourcesPath + "architecture.json", "w") as jsonFile:
            jsonFile.write(self.model.to_json())
        print("Saved model architecture to disk.")

        self.model.save(filepath=self.resourcesPath + "model_weights.h5", overwrite=True)
        print("Saved model weights to disk.")

        np.save(self.resourcesPath + 'classes.npy', self.encoder.classes_)
        np.save(self.resourcesPath + 'y_type_.npy', self.encoder.y_type_)
        np.save(self.resourcesPath + 'sparse_input_.npy', self.encoder.sparse_input_)
        print("Saved encoded classes to disk.")
        return self

    def train(self):
        # Load the dataset and split it
        train_images, train_labels = self.dataset.load()
        number_of_classes = self.dataset.get_number_of_labels()

        start = int(self.trainPercentage * len(train_labels))
        end = len(train_labels)
        valid_images = train_images[start:end]
        valid_labels = train_labels[start:end]
        train_images = train_images[0:start]
        train_labels = train_labels[0:start]

        # Creating Tensors
        train_images = train_images.reshape((len(train_images), 40, 40, 1))
        train_images = train_images.astype('float32') / 255  # converting to a [0, 1] scale
        valid_images = valid_images.reshape((len(valid_images), 40, 40, 1))
        valid_images = valid_images.astype('float32') / 255  # converting to a [0, 1] scale
        # test_images = test_images.reshape((10000, 28, 28, 1))
        # test_images = test_images.astype('float32') / 255  # converting to a [0, 1] scale

        # One-hot encode labels
        self.encoder = LabelBinarizer()
        encoded_train_labels = self.encoder.fit_transform(train_labels)
        encoded_valid_labels = self.encoder.fit_transform(valid_labels)

        # ------------
        # CNN Training
        # ------------
        model = self.__build_model(number_of_classes)
        model.summary()
        model.compile(
            optimizers.sgd(lr=0.2, decay=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(
            x=train_images,
            y=encoded_train_labels,
            batch_size=2500,
            epochs=10,
            validation_data=(valid_images, encoded_valid_labels)
        )
        self.model = model
        return self

    def __build_model(self, number_of_labels):
        # ----------------
        # CNN Architecture
        # ----------------
        model = models.Sequential()

        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), use_bias=True, input_shape=(40, 40, 1)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.01))

        model.add(Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), use_bias=True))
        model.add(Activation('relu'))
        model.add(Dropout(rate=0.01))
        model.add(Conv2D(filters=36, kernel_size=(3, 3), strides=(1, 1), use_bias=True))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.01))
        model.add(layers.Flatten())

        model.add(Dense(units=512, use_bias=True))
        model.add(Activation('relu'))
        model.add(Dense(units=number_of_labels, use_bias=True))
        model.add(Activation('softmax'))

        return model
