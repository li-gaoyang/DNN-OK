import pandas as pd
from six import b

import tensorflow as tf
from tensorflow.keras import layers
import os
# from keras.constraints import MinMaxNorm
# from keras.constraints import min_max_norm
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Conv2D,LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import os
import shutil
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


global x_callback
global y_callback


class CustomCallback(keras.callbacks.Callback):
    # 定义时需要注意 这里一些logs可能是空 所以需要注意拿的元素是否为None
    # example:
    # Training: end ob f batch 0; got log keys: ['batch', 'size', 'loss', 'mean_absolute_error']
    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model
    def on_train_begin(self, logs=None): 
        keys = list(logs.keys())
        # print("Starting training; got log keys: {}".format(keys))
 
    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        # print("Stop training; got log keys: {}".format(keys))
 
    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        # print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
 
    def on_epoch_end(self, epoch, logs=None):  
        with tf.GradientTape() as tape:
            tape.watch(x_callback)
            y = self.model(x_callback)
            dy1_dx = tape.gradient(target=y, sources=x_callback)  # 求梯度,
            # print(dy1_dx)
            for i2 in dy1_dx:
                tt = i2.numpy()[0]
                if tt < -1:
                    self.model.stop_training = True
                    break
                if r2_score(y_callback,y)>0.999:
                    self.model.stop_training = True
                    break


    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        # print("Start testing; got log keys: {}".format(keys))
 
    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        # print("Stop testing; got log keys: {}".format(keys))
 
    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        # print("Start predicting; got log keys: {}".format(keys))
 
    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        # print("Stop predicting; got log keys: {}".format(keys))
 
    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        # print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
 
    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        # print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
 
    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        # print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))
 
    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        # print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))
 
    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        # print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))
 
    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        # print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


def build_model():
    model = keras.Sequential()
    #activation="relu"
    activation='softplus' # 
    kernel_initializer="he_normal" #lecun_uniform lecun_normal
    #kernel_initializer=keras.initializers.Ones()
    #kernel_initializer="glorot_u7niform" glorot_normal he_uniform  he_normal #不能用，坚决不能用
    #kernel_initializer=""
    #loss='mean_squared_logarithmic_error'
    loss="mean_squared_logarithmic_error" # mean_squared_error mean_absolute_error mean_squared_logarithmic_error mean_absolute_percentage_error
    learning_rate=0.0001
    optimizer=optimizers.Adam(learning_rate=learning_rate)
    #optimizer='mae' #mae RMSprop Adam learning_rate=0.001
    model.add(layers.Dense(1, activation='softplus',kernel_initializer=keras.initializers.Ones()))
    model.add(layers.Dense(2, activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(4, activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(8, activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(16, activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(32,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(64,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(128,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(256,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(256,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(128,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(64,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(32,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(16,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(8, activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(4, activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(2,activation=activation,kernel_initializer=kernel_initializer))
    model.add(layers.Dense(1,activation='softplus',kernel_initializer=keras.initializers.Ones()))
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc','mae'])

    return model


def trains(dx, dy):
    shutil.rmtree('img')
    os.mkdir('img')
    dx2 = []
    for i in dx:
        temp1 = []
        temp1.append(i)
        dx2.append(temp1)

    dy2 = []
    for i in dy:
        temp1 = []
        temp1.append(i)
        dy2.append(temp1)

    train_X = np.array(dx2)
    train_Y = np.array(dy2)
 
    x = tf.convert_to_tensor(train_X, tf.double, name='t')

    # 拟合很多个模型，直到拟合的模型为非增函数为止，选取前一个增函数（如果用tf自己写原生的梯度下降函数会快点）
    model=build_model()
 
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(train_X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_X[train_index], train_X[test_index]
        y_train, y_test = train_Y[train_index], train_Y[test_index]
        global x_callback
        global y_callback
        y_callback=y_train
        x_callback = tf.convert_to_tensor(X_train, tf.double, name='t')
        model.fit(X_train, y_train, epochs=1000,validation_data=(X_test,y_test),callbacks=[CustomCallback()],verbose=1 )

    # test_Y = model.predict(train_X) 
    # s1 = plt.scatter(train_X, train_Y)
    # s2 = plt.plot(train_X, test_Y)
    # plt.legend([s1, s2[0]], ['Accuracy', 'OOVs'])
    # plt.show()
       
    return model

if __name__ == "__main__":
    dx = [
        10.52631579,
        12.52631579,
        15.78947368,
        18.78947368,
        21.05263158,
        23.05263158, 
        26.31578947, 
        31.57894737, 
        36.84210526, 
        42.10526316, 
        47.36842105, 
        52.63157895,
        57.89473684, 
        63.15789474, 
        68.42105263, 
        73.68421053, 
        78.94736842, 
        84.21052632, 
        89.47368421, 
        94.73684211, 
        100]

    dy = [
        381.86111111,
        401.86111111,  
        495.51234568, 
        555.51234568, 
        657.22991071, 
        747.22991071, 
        810.61328125, 
        829.56505102, 
        959.47321429, 
        1193.03354633, 
        1278.13432836, 
        1387.66911765,
        1342.26541096, 
        1397.23214286, 
        1450.60606061, 
        1437.80434783, 
        1467.28787879, 
        1526.42216981, 
        1698.65865385, 
        1554.27027027, 
        1609.52702703]

    trains(dx, dy)
