import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16

from keras.utils import to_categorical

class ConvolutionNeuralNetwork:

    def __init__(self, num_epochs=30,batch_size = 16, layers=2, dropout=0.5):
        self.num_epochs = num_epochs
        self.layers = layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        self.n_labels = 7

        self.label_path = os.path.join("trained_models", "CNN_LABEL_ENCODER.pkl")
        self.model_path = os.path.join("trained_models", "CNN_MODEL.keras")

        os.makedirs("trained_models", exist_ok=True)

        if os.path.exists(self.model_path) and os.path.exists(self.label_path):
            self.load()
        else:
            self.model = self.build_model()
            self.train()

    def build_model(self):
        vgg_expert = VGG16(weights = 'imagenet', include_top = False, input_shape = (64, 64, 3))

        vgg_model = Sequential()
        vgg_model.add(vgg_expert)

        vgg_model.add(GlobalAveragePooling2D())
        vgg_model.add(Dense(1024, activation = 'relu'))
        vgg_model.add(Dropout(0.3))
        vgg_model.add(Dense(512, activation = 'relu'))
        vgg_model.add(Dropout(0.3))
        vgg_model.add(Dense(self.n_labels, activation = 'sigmoid'))
        
        vgg_model.compile(loss = 'categorical_crossentropy',
          optimizer = SGD(learning_rate=0.001, momentum=0.95),
          metrics=['accuracy'])

        return vgg_model

    def load_data(self, path: str):
        loaded = np.load(path)
        data = loaded['data']
        labels = loaded['labels']
        return data, labels

    # def load_and_preprocess(self):
    #     data, labels = self.load_data("datasets/emotions_dataset_64_64.npz")

    #     data = data.reshape((-1, 64, 64, 1))

    #     X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    #     y_train_encoded = self.label_encoder.fit_transform(y_train)
    #     y_test_encoded = self.label_encoder.transform(y_test)

    #     y_train_onehot = to_categorical(y_train_encoded)
    #     y_test_onehot = to_categorical(y_test_encoded)

    #     return X_train, X_test, y_train_encoded, y_test_encoded, y_train_onehot, y_test_onehot
    
    def load_and_preprocess(self):
        data, labels = self.load_data("datasets/emotions_dataset_64_64.npz")

        data = data.reshape((-1, 64, 64, 1))
        data = np.repeat(data, 3, axis=-1)  # simulate RGB

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        y_train_onehot = to_categorical(y_train_encoded, num_classes=self.n_labels)
        y_test_onehot = to_categorical(y_test_encoded, num_classes=self.n_labels)

        return X_train, X_test, y_train_encoded, y_test_encoded, y_train_onehot, y_test_onehot


    def train(self):
        X_train, X_test, y_train_encoded, y_test_encoded, y_train_onehot, y_test_onehot = self.load_and_preprocess()

        self.history = self.model.fit(
            X_train,
            y_train_onehot,
            validation_data=(X_test, y_test_onehot),
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            verbose=1,
            shuffle=True
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
        X = X.astype('float64') / 255.0
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=0)
        if X.shape[-1] != 1:
            X = X.reshape((-1, 64, 64, 1))
        return self.model.predict(X)

    def score(self, X, y_true_encoded):
        y_pred_probs = self.predict(X)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        return accuracy_score(y_true_encoded, y_pred_labels)


    def predict_label(self, image_array):
        image_array = image_array.astype('float64') / 255.0
        image_array = image_array.reshape((1, 64, 64, 1))
        probs = self.model.predict(image_array)
        pred_index = np.argmax(probs, axis=1)[0]
        return self.label_encoder.inverse_transform([pred_index])[0]


cnn = ConvolutionNeuralNetwork(num_epochs=20)
# cnn.predict_label()