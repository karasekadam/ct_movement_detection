import os

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import pandas as pd


# Define the CNN architecture for each image
def create_shared_cnn():
    inputs = Input(shape=(1650, 1250, 1))  # Input image size (height, width, channels)

    # Convolutional layers with downsampling (MaxPooling) and strides
    x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(inputs)  # 1650x1250 -> 825x625
    x = Dropout(0.2)(x)  # Dropout layer to prevent overfitting
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Further downsample -> 412x312
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)  # 412x312 -> 206x156
    x = Dropout(0.2)(x)  # Dropout layer to prevent overfitting
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Further downsample -> 103x78
    x = Flatten()(x)  # Flatten feature maps

    # Create the CNN model
    model = Model(inputs, x)
    return model


def create_compare_model():
    # Create the shared CNN model
    shared_cnn = create_shared_cnn()

    # Define two input images
    input_img1 = Input(shape=(1650, 1250, 1))  # First image
    input_img2 = Input(shape=(1650, 1250, 1))  # Second image

    # Apply the shared CNN to both inputs
    features_img1 = shared_cnn(input_img1)
    features_img2 = shared_cnn(input_img2)

    # Concatenate the extracted features from both images
    combined_features = Concatenate()([features_img1, features_img2])

    # Fully connected layers after concatenation
    x = Dense(128, activation='relu')(combined_features)
    x = Dense(1, activation='sigmoid')(x)  # Sigmoid output for binary classification

    # Create the final model
    model = Model(inputs=[input_img1, input_img2], outputs=x)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])

    # Summary of the model architecture
    model.summary()

    return model


def create_dif_model():
    # Create the shared CNN model
    shared_cnn = create_shared_cnn()

    input_img1 = Input(shape=(1650, 1250, 1))  # First image
    features_img1 = shared_cnn(input_img1)
    # Fully connected layers after concatenation
    x = Dense(128, activation='relu')(features_img1)
    x = Dense(1, activation='sigmoid')(x)  # Sigmoid output for binary classification

    model = Model(inputs=input_img1, outputs=x)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    # Summary of the model architecture
    model.summary()

    return model


# Custom data generator for two-image input
class MovementCompareDataGenerator(Sequence):
    def __init__(self, image_paths1, image_paths2, labels, batch_size, image_dir, image_size=(1650, 1250)):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.image_dir = image_dir
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        batch_img1 = self.image_paths1[index * self.batch_size:(index + 1) * self.batch_size]
        batch_img2 = self.image_paths2[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        # Preprocess images and labels
        img1, img2, labels = self.__data_generation(batch_img1, batch_img2, batch_labels)
        return (img1, img2), labels

    def __data_generation(self, batch_img1, batch_img2, batch_labels):
        img1 = np.array([self.load_image(self.image_dir + file + ".tiff") for file in batch_img1])
        img2 = np.array([self.load_image(self.image_dir + file + ".tiff") for file in batch_img2])
        labels = np.array(batch_labels)

        return img1, img2, labels

    def load_image(self, filepath):
        image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
        image = np.transpose(image, (1, 0))  # Transpose to (height, width)
        # image = cv2.resize(image, self.image_size)  # Resize to 1650x1250
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = image / 65535  # Normalize to [0, 1]
        return image


class MovementDifDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, image_dir, image_size=(1650, 1250)):
        self.image_paths = image_paths
        self.image_dir = image_dir
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        batch_img = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        # Preprocess images and labels
        img, labels = self.__data_generation(batch_img, batch_labels)
        return img, labels

    def __data_generation(self, batch_img, batch_labels):
        img = np.array([self.load_image(self.image_dir + file + ".tiff") for file in batch_img])
        labels = np.array(batch_labels)
        return img, labels

    def load_image(self, filepath):
        image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
        image = np.transpose(image, (1, 0))  # Transpose to (height, width)
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        return image


def compare_model_experiment():
    labels_df = pd.read_excel("labeled_dataset/dataset_labels_noise.xlsx")
    image_paths1 = labels_df['img1'].tolist()
    image_paths2 = labels_df['img2'].tolist()
    labels = labels_df['label'].tolist()
    # Create the data generator
    train_generator = MovementCompareDataGenerator(image_paths1, image_paths2, labels, image_dir="labeled_dataset/grayscale/", batch_size=1)
    print(train_generator.__len__())
    print(train_generator.__getitem__(0))
    model = create_compare_model()
    # Train the model
    model.fit(train_generator, epochs=100)


def dif_model_experiment():
    # Example paths to images and labels
    labels_df = pd.read_excel("labeled_dataset/dataset_labels_noise.xlsx")
    labels_df["image_dif"] = labels_df["img1"] + "_dif_" + labels_df["img2"]
    dif_images_paths = sorted(labels_df['image_dif'].tolist())
    labels = labels_df['label'].tolist()

    dif_images_train, dif_images_val, labels_train, labels_val = train_test_split(dif_images_paths, labels,
                                                                                  test_size=0.3, random_state=42, shuffle=True)

    train_generator = MovementDifDataGenerator(dif_images_train, labels_train, image_dir="labeled_dataset/dif/", batch_size=1)
    val_generator = MovementDifDataGenerator(dif_images_val, labels_val, image_dir="labeled_dataset/dif/", batch_size=1)
    model = create_dif_model()
    model.fit(train_generator, epochs=30, validation_data=val_generator)
    model.save("dif_model.keras")


def show_wrongly_dif_classified(model_path: str, labels_path: str, images_dir_path: str) -> None:
    labels_df = pd.read_excel(labels_path)
    labels_df["image_dif"] = labels_df["img1"] + "_dif_" + labels_df["img2"]
    dif_images_paths = labels_df['image_dif'].tolist()
    labels = labels_df['label'].tolist()

    model = models.load_model(model_path)

    for i in range(len(dif_images_paths)):
        img = cv2.imread(images_dir_path + dif_images_paths[i] + ".tiff", cv2.IMREAD_ANYDEPTH)
        img = np.transpose(img, (1, 0))
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)[0][0]
        rounded_pred =  round(model.predict(img)[0][0], 0)
        if rounded_pred != labels[i]:
            print(f"Image: {dif_images_paths[i]}, Predicted: {pred}, Actual: {labels[i]}")


dif_model_experiment()
show_wrongly_dif_classified("dif_model.keras", "labeled_dataset/test_dataset_labels.xlsx", "labeled_dataset/dif_test/")
