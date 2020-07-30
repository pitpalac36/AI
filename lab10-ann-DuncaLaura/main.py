import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


def convert_to_sepia(path):
    img = Image.open(path)
    width, height = img.size
    pixels = img.load()
    for py in range(height):
        for px in range(width):
            r, g, b = img.getpixel((px, py))
            newr = int(0.393 * r + 0.769 * g + 0.189 * b)
            newg = int(0.349 * r + 0.686 * g + 0.168 * b)
            newb = int(0.272 * r + 0.534 * g + 0.131 * b)
            if newr > 255:
                newr = 255
            if newg > 255:
                newg = 255
            if newb > 255:
                newb = 255
            pixels[px, py] = (newr, newg, newb)
    return img


def create_sepia_base():
    path = 'before_sepia_train/'
    entries = os.listdir(path)
    for each in entries:
        img = convert_to_sepia(path + each.title())
        img.show()
        img.save('data/train_set/sepia/filtered' + each.title(), 'jpeg')

    path = 'before_sepia_test/'
    entries = os.listdir(path)
    for each in entries:
        img = convert_to_sepia(path + each.title())
        img.show()
        img.save('data/test_set/sepia/filtered' + each.title(), 'jpeg')

    img = convert_to_sepia(path + 'single_prediction/guess')
    img.show()
    img.save('single_prediction/Guess.Jpeg', 'jpeg')


def main():
    classifier = Sequential()

    # convolution step
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # pooling step
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # flattening step
    classifier.add(Flatten())

    # connect layers
    classifier.add(Dense(units=128, activation='relu'))

    # initialise output layer
    classifier.add(Dense(units=1, activation='sigmoid'))

    # compile CNN model
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # synthetise data
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('data/train_set',
                                                     target_size=(64, 64),
                                                     batch_size=10,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('data/test_set',
                                                target_size=(64, 64),
                                                batch_size=10,
                                                class_mode='binary')

    classifier.fit_generator(training_set,
                             steps_per_epoch=100,
                             epochs=10,
                             validation_data=test_set,
                             validation_steps=20)

    test_image = image.load_img('data/single_prediction/Guess.Jpeg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        prediction = 'sepia'
    else:
        prediction = 'not_sepia'
    print("Guess is " + prediction)


# create_sepia_base()
main()
