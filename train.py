# Created by Oumaima Souli.

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# Building the classifier model based on dataset
# for better prediction we used the dataset provided in https://data-flair.training/blogs/download-face-mask-data/
def train():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary')

    test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary')

    saved_model = model.fit_generator(
        training_set,
        epochs=10,
        validation_data=test_set,
    )

    model.save('model.h5', saved_model)

# train()