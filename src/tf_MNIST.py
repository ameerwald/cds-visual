# generic tools
import numpy as np
# tools from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
# matplotlib
import matplotlib.pyplot as plt

def load_data():
    # downloading the dataset, and separate X and y and keeping the dataset in full as 'data'
    data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    # normalise data
    data = data.astype("float")/255.0
    # split data
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                        labels, 
                                                        test_size=0.2)
    # convert labels to one-hot encoding
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return X_train, X_test, y_train, y_test, lb

def train_model(X_train, y_train):
    # define architecture 784x256x128x10
    model = Sequential() 
    model.add(Dense(256,  
                    input_shape=(784,), 
                    activation="relu")) 
    model.add(Dense(128,  
                    activation="relu"))
    model.add(Dense(10, 
                    activation="softmax")) 
    model.summary()
    # train model using SGD
    sgd = SGD(0.01) 
    model.compile(loss="categorical_crossentropy", 
                optimizer=sgd, 
                metrics=["accuracy"])

    history = model.fit(X_train, y_train, 
                        epochs=10, 
                        batch_size=32)
    return model, history

# show classification report of how the model does 
def predict(X_test, y_test, model, lb):
    # evaluate network
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(y_test.argmax(axis=1), 
                                predictions.argmax(axis=1), 
                                target_names=[str(x) for x in lb.classes_]))

def main():
    # load the data
    X_train, X_test, y_train, y_test, lb = load_data()
    # define and train the model 
    model, history = train_model(X_train, y_train)
    # show classification report for the model 
    predict(X_test, y_test, model, lb)

if __name__=="__main__":
    main()