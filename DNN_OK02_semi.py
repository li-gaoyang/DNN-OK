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
from tensorflow.keras.layers import Activation, Conv2D
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from tensorflow.keras.utils.generic_utils import get_custom_objects

# sill=1
# nug=80

# def swish(x):
#     '''
#     x：每个值
#     '''
#     return (sill+nug*K.sigmoid(x))


# get_custom_objects().update({'swish': Activation(swish)})


def trains(dx, dy):

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

    model = keras.Sequential()

    model.add(layers.Dense(8, activation='relu',
              kernel_initializer=keras.initializers.ones()))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(128,activation='relu'))
    
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(8,activation='relu'))
    model.add(layers.Dense(1, activation='relu',
              kernel_initializer=keras.initializers.ones()))

    # loss=keras.losses.MAE()
    opt = optimizers.Adam(learning_rate=0.01)
    # mse 对离群点权重赋值较高（需要求均值）
    # mae 对损失适用于训练数据被离群点损坏的时候（求中位数）
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])
    # callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1)#使用loss作为监测数据，轮数设置为1
    x = tf.convert_to_tensor(train_X, tf.double, name='t')

    # 拟合很多个模型，直到拟合的模型为非增函数为止，选取前一个增函数（如果用tf自己写原生的梯度下降函数会快点）

    flag = 0
    i = 30
    while flag == 0:
        i = i+1
        if i % 100== 0:  # 每500次就检查一次这个DNN拟合的函数是否为增函数。，如果非增，就选择上一次检查过的增函数，并中断
            model.fit(train_X, train_Y, epochs=i, verbose=2)
            x = tf.convert_to_tensor(train_X, tf.double, name='t')

            with tf.GradientTape() as tape:
                tape.watch(x)
                y = model(x)
                dy1_dx = tape.gradient(target=y, sources=x)  # 求梯度,
                print(dy1_dx)
                if np.mean(dy1_dx) == 0:  # 防止梯度消失
                    flag = 1

                fig = plt.figure()
                # 新建子图1
                ax1 = fig.add_subplot()
                f1 = ax1.scatter(train_X, train_Y)

                f2 = ax1.plot(x, y, c="r")
                ax1.legend([f1, f2[0]], ['points', 'DNN'])
                plt.savefig(str(i)+".png")

                for i2 in dy1_dx:
                    tt = i2.numpy()[0]
                    if tt < 0:
                        flag = 1
                        break
            if flag == 0:
                model.save('model_DNN.h5')

    # model.summary()#数据控制台
    model_res = load_model("model_DNN.h5")
    test_Y = model_res.predict(train_X)
    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(train_X, tf.double, name='t')

        tape.watch(x)
        y = model_res(x)
        dy1_dx = tape.gradient(target=y, sources=x)  # 求梯度,
        print(dy1_dx)

   

    # s1 = plt.scatter(train_X, train_Y)
    # s2 = plt.plot(train_X, test_Y)
    # plt.legend([s1, s2[0]], ['Accuracy', 'OOVs'])
    # plt.show()
    return model_res


if __name__ == "__main__":
    dx = [10.52631579, 15.78947368, 21.05263158, 26.31578947, 31.57894737, 36.84210526, 42.10526316, 47.36842105, 52.63157895,
          57.89473684, 63.15789474, 68.42105263, 73.68421053, 78.94736842, 84.21052632, 89.47368421, 94.73684211, 100]

    dy = [381.86111111, 495.51234568, 657.22991071, 810.61328125, 829.56505102, 959.47321429, 1193.03354633, 1278.13432836, 1387.66911765,
          1342.26541096, 1397.23214286, 1450.60606061, 1437.80434783, 1467.28787879, 1526.42216981, 1698.65865385, 1554.27027027, 1609.52702703]

    trains(dx, dy)
