import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import (
    Dense, Conv2D, InputLayer, Activation, MaxPooling2D,
    Dropout, Flatten
)
from keras.utils import to_categorical

class ConvolutionNeuralNetwork:

    def __init__(self, num_epochs=30, layers=2, dropout=0.5):
        self.num_epochs = num_epochs
        self.layers = layers
        self.dropout = dropout
        self.label_encoder = LabelEncoder()

        self.label_path = os.path.join("trained_models", "CNN_LABEL_ENCODER.pkl")
        self.model_path = os.path.join("trained_models", "CNN_MODEL.keras")

        os.makedirs("trained_models", exist_ok=True)

        if os.path.exists(self.model_path) and os.path.exists(self.label_path):
            self.load()
        else:
            self.model = self.build_model()
            self.train()

    def build_model(self):
        model = Sequential()

        model.add(InputLayer(input_shape=(32, 32, 1)))

        for _ in range(self.layers):
            model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(self.dropout))
            


        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(7, activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def load_data(self, path: str):
        loaded = np.load(path)
        data = loaded['data']
        labels = loaded['labels']
        return data, labels

    def load_and_preprocess(self):
        data, labels = self.load_data("datasets/emotions_dataset_32_32.npz")

        data = data.reshape((-1, 32, 32, 1))

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        y_train_onehot = to_categorical(y_train_encoded)
        y_test_onehot = to_categorical(y_test_encoded)

        return X_train, X_test, y_train_encoded, y_test_encoded, y_train_onehot, y_test_onehot

    def train(self):
        X_train, X_test, y_train_encoded, y_test_encoded, y_train_onehot, y_test_onehot = self.load_and_preprocess()

        self.history = self.model.fit(
            X_train,
            y_train_onehot,
            validation_data=(X_test, y_test_onehot),
            epochs=self.num_epochs,
            batch_size=32,
            verbose=2
        )

        accuracy = self.score(X_test, y_test_encoded)
        print(f"Test Accuracy: {accuracy:.4f}")
        self.save()
        return accuracy

    def save(self, model_path=None, label_path=None):
        model_path = model_path or self.model_path
        label_path = label_path or self.label_path
        self.model.save(model_path)
        joblib.dump(self.label_encoder, label_path)

    def load(self, model_path=None, label_path=None):
        model_path = model_path or self.model_path
        label_path = label_path or self.label_path
        self.model = keras.models.load_model(model_path)
        self.label_encoder = joblib.load(label_path)

    def predict(self, X):
        X = X.astype('float32') / 255.0
        if len(X.shape) == 3:  # single image, no batch dim
            X = np.expand_dims(X, axis=0)
        if X.shape[-1] != 1:  # ensure single channel
            X = X.reshape((-1, 32, 32, 1))
        return self.model.predict(X)

    def score(self, X, y_true_encoded):
        y_pred_probs = self.predict(X)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        return accuracy_score(y_true_encoded, y_pred_labels)

    def predict_label(self, image_array):
        image_array = image_array.astype('float32') / 255.0
        image_array = image_array.reshape((1, 32, 32, 1))
        probs = self.model.predict(image_array)
        pred_index = np.argmax(probs, axis=1)[0]
        return self.label_encoder.inverse_transform([pred_index])[0]


cnn = ConvolutionNeuralNetwork(num_epochs=20)
# cnn.predict_label()