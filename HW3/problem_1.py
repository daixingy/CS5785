import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD


print(keras.__version__)
print(tf.__version__)

from keras.datasets import mnist

def show_example(data):
    plt.imshow(data[0])
    plt.savefig("./example.png")
    

def preprocess(data, width, length):
    outdata = []
    for i in range(len(data)):
        outdata.append(data[i].reshape((width, length, -1))/255.0)
    return np.array(outdata)

def one_hot(data):
    return to_categorical(data)

def modified_cnn():
    model = Sequential()
    # Convolution layer
    model.add(
            Conv2D(32 , (3, 3),
            activation ='relu',
            kernel_initializer = 'he_uniform',
            input_shape=(28,28,1))
            )
    # Maxpooling layer
    model.add(MaxPooling2D((2 , 2)))
    model.add(
            Conv2D(64, (3, 3),
            activation = 'relu',
            kernel_initializer = 'he_uniform',
            input_shape = (13, 13, 32))
            )
    model.add(MaxPooling2D((2 , 2)))
    model.add(Flatten())
    # Dense layer of 100 neurons
    model.add(
            Dense(100 ,
            activation = 'relu',
            kernel_initializer = 'he_uniform')
            )
    model.add(Dense(10 ,activation='softmax'))
    # initialize optimizer
    opt = SGD(lr=0.01 , momentum=0.9)
    # compile model
    model.compile(
            optimizer = opt,
            loss ='categorical_crossentropy',
            metrics =['accuracy']
            )
    return model


def create_cnn() :
    # define using Sequential
    model = Sequential()
    # Convolution layer
    model.add(
            Conv2D(32 , (3, 3),
            activation ='relu',
            kernel_initializer = 'he_uniform',
            input_shape=(28,28,1))
            )
    # Maxpooling layer
    model.add(MaxPooling2D((2 , 2)))
    # Flatten output
    model.add(Flatten())
    # Dense layer of 100 neurons
    model.add(
            Dense(100 ,
            activation = 'relu',
            kernel_initializer = 'he_uniform')
            )
    model.add(Dense(10 ,activation='softmax'))
    # initialize optimizer
    opt = SGD(lr=0.01 , momentum=0.9)
    # compile model
    model.compile(
            optimizer = opt,
            loss ='categorical_crossentropy',
            metrics =['accuracy']
            )
    return model

def cnn(lr=0.01) :
    # define using Sequential
    model = Sequential()
    # Convolution layer
    model.add(
            Conv2D(32 , (3, 3),
            activation ='relu',
            kernel_initializer = 'he_uniform',
            input_shape=(28,28,1))
            )
    # Maxpooling layer
    model.add(MaxPooling2D((2 , 2)))
    model.add(
            Conv2D(64, (3, 3),
            activation = 'relu',
            kernel_initializer = 'he_uniform',
            input_shape = (13, 13, 32))
            )
    model.add(MaxPooling2D((2 , 2)))
    # Flatten output
    model.add(Flatten())
    # Dense layer of 100 neurons
    model.add(
            Dense(100 ,
            activation = 'relu',
            kernel_initializer = 'he_uniform')
            )
    model.add(Dense(10 ,activation='softmax'))
    # initialize optimizer
    opt = SGD(lr=lr , momentum=0.9)
    # compile model
    model.compile(
            optimizer = opt,
            loss ='categorical_crossentropy',
            metrics =['accuracy']
            )
    return model

def print_layers(model):
    for i in model.layers:
        print(i)

def train_and_evaluate(model, train_X, train_Y, test_X, test_Y):
    epoch_history = model.fit(train_X, train_Y, batch_size=32, epochs=10, validation_split=0.1)
    score = model.evaluate(test_X, test_Y, verbose=0)
    # print(epoch_history.history['accuracy'])
    print(score)
    return model

def experiment(model, train_X, train_Y, test_X, test_Y):
    training_history = []
    validation_history = []
    for i in range(5):
        epoch_history = model.fit(train_X, train_Y, batch_size=32, epochs=10, validation_split=0.1)
        score = model.evaluate(test_X, test_Y, verbose=0)
        # recode the last accuracy
        training_history.append(epoch_history.history['accuracy'][-1])
        validation_history.append(epoch_history.history['val_accuracy'][-1])
    X_axies = [10, 20, 30, 40, 50]
    plt.figure()
    l1, = plt.plot(X_axies, training_history, color='red')
    l2, = plt.plot(X_axies, validation_history, color='blue')
    plt.legend(handles=[l1,l2],labels=['training acc','validation acc'],loc='best')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title("experimentation")
    plt.savefig("./experimentation.png")

def dropout_model():
    # define using Sequential
    model = Sequential()
    # Convolution layer
    model.add(
            Conv2D(32 , (3, 3),
            activation ='relu',
            kernel_initializer = 'he_uniform',
            input_shape=(28,28,1))
            )
    # Maxpooling layer
    model.add(MaxPooling2D((2 , 2)))
    # Flatten output
    model.add(Flatten())
    # Dense layer of 100 neurons
    model.add(Dropout(0.5))
    model.add(
            Dense(100 ,
            activation = 'relu',
            kernel_initializer = 'he_uniform')
            )
    model.add(Dense(10 ,activation='softmax'))
    # initialize optimizer
    opt = SGD(lr=0.01 , momentum=0.9)
    # compile model
    model.compile(
            optimizer = opt,
            loss ='categorical_crossentropy',
            metrics =['accuracy']
            )
    return model


if __name__ == "__main__":
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    train_X = preprocess(train_X, 28, 28)
    test_X = preprocess(test_X, 28, 28)
    train_Y = one_hot(train_Y)
    test_Y = one_hot(test_Y)
    # model = create_cnn()
    # d_model = dropout_model()
    # print_layers(model)
    # model = train_and_evaluate(model, train_X, train_Y, test_X, test_Y)
    # experiment(model, train_X, train_Y, test_X, test_Y)
    # m_model = modified_cnn()
    # m_model = train_and_evaluate(m_model, train_X, train_Y, test_X, test_Y)
    # iv_model_1 = cnn(0.001)
    # iv_model_2 = cnn(0.1)
    # train_and_evaluate(iv_model_1, train_X, train_Y, test_X, test_Y)
    # train_and_evaluate(iv_model_2, train_X, train_Y, test_X, test_Y)

    

    

