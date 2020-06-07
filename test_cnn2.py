# Marisetti, Mohith

import pytest
import numpy as np
from cnn import CNN
import os
import tensorflow.keras as keras
import os.path
from tensorflow.keras.models import Model
from os import path
from tensorflow.keras.datasets import cifar10

def test_train():

    my_cnn = CNN()
    my_cnn.add_input_layer(shape=(784,),name="input")
    my_cnn.append_dense_layer(num_nodes=64,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=64,activation="relu",name="dense2")
    my_cnn.append_dense_layer(num_nodes=10,activation="softmax",name="predictions1")
    my_cnn.append_dense_layer(num_nodes=5,activation="relu",name="predictions2")
    my_cnn.append_dense_layer(num_nodes=10,activation="softmax",name="predictions3")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')


    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    my_cnn.set_loss_function(loss="SparseCategoricalCrossentropy")
    my_cnn.set_optimizer(optimizer="RMSprop")
    my_cnn.set_metric(metric="accuracy")



    history=my_cnn.train(x_train, y_train, batch_size=64, num_epochs=3)

    assert np.allclose(history.history['loss'],np.array([0.8855341824245453, 0.3022494815063477, 0.2548141068649292] ),rtol=1,atol=0.05)
    assert np.allclose(history.history['sparse_categorical_accuracy'],np.array([0.6689, 0.93858, 0.94988]),rtol=1,atol=0.05)


def test_evalute():

    my_cnn=CNN()
    my_cnn.add_input_layer(shape=(784,),name="input")
    my_cnn.append_dense_layer(num_nodes=64,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=64,activation="relu",name="dense2")
    my_cnn.append_dense_layer(num_nodes=10,activation="softmax",name="predictions1")
    my_cnn.append_dense_layer(num_nodes=5,activation="relu",name="predictions2")
    my_cnn.append_dense_layer(num_nodes=10,activation="softmax",name="predictions3")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    my_cnn.set_loss_function(loss="SparseCategoricalCrossentropy")
    my_cnn.set_optimizer(optimizer="RMSprop")
    my_cnn.set_metric(metric="accuracy")



    history=my_cnn.train(x_train, y_train, batch_size=64, num_epochs=3)

    results = my_cnn.evaluate(x_test, y_test)

    assert np.allclose(results,np.array( [0.24014270609021188, 0.947] ),rtol=1,atol=0.1)
