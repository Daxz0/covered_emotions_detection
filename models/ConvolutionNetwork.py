import pandas as pd
import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras # type: ignore
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class ConvolutionNeuralNetwork:
    
    def __init__(self, num_epochs=30, layers=4, dropout=0.5):
        self.num_epochs = num_epochs
        self.layers = layers
        self.dropout = dropout
        self.label_encoder = LabelEncoder()
        self.label_path = os.path.join("trained_models", "CNN_LABEL_ENCODER.pkl")
        self.model_path = os.path.join("trained_models", "CNN_MODEL.keras")
        self.history = None

        
        if os.path.exists(self.model_path) and os.path.exists(self.label_path):
            self.load()
        else:
            self.model = self.build_model()
            self.train()

    def build_model(self):
        model = Sequential()
        model.add(Reshape((32, 32, 3)))

        for _ in range(self.layers):
            model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
            model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(7))
        model.add(Activation('softmax'))

        opt = keras.optimizers.RMSprop(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def train(self):
        X_train, X_test, y_train_encoded, y_test_encoded, y_train_onehot, y_test_onehot = self.load_and_preprocess()
        self.history = self.model.fit(X_train, y_train_onehot, epochs=self.num_epochs, batch_size=10, verbose=2) # type: ignore
        accuracy = self.score(X_test, y_test_encoded)
        self.save()
        return accuracy

    def save(self, model_path=None, label_path=None):
        model_path = model_path or self.model_path
        label_path = label_path or self.label_path
        self.model.save(model_path) # type: ignore
        joblib.dump(self.label_encoder, label_path)

    def load(self, model_path=None, label_path=None):
        model_path = model_path or self.model_path
        label_path = label_path or self.label_path
        self.model = keras.models.load_model(model_path)
        print(self.model)
        self.label_encoder = joblib.load(label_path)

    def predict(self, X):
        return self.model.predict(X) # type: ignore

    def score(self, X, y_true_encoded):
        y_pred_probs = self.predict(X)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        return accuracy_score(y_true_encoded, y_pred_labels)

    def predict_label(self, image_array):
        image_array = np.expand_dims(image_array, axis=0)
        probs = self.model.predict(image_array)
        pred_index = np.argmax(probs, axis=1)[0]
        return self.label_encoder.inverse_transform([pred_index])[0]

