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

import os
import shutil
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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
    #par=GCV(train_X,train_Y)
  
    model = keras.Sequential()
    #activation="relu"
    activation='softplus' # 
    kernel_initializer="he_normal" #lecun_uniform lecun_normal
    #kernel_initializer=keras.initializers.Ones()
    #kernel_initializer="glorot_uniform" glorot_normal he_uniform  he_normal #不能用，坚决不能用
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
    # model.add(layers.Dense(8, activation=par['activation'],kernel_initializer=keras.initializers.Zeros()))
    # model.add(layers.Dense(16, activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(32,activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(64,activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(128,activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(512,activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(128,activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(64,activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(32,activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(16,activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(8,activation=par['activation'],kernel_initializer=par['init_mode']))
    # model.add(layers.Dense(1,activation=par['activation'],kernel_initializer=keras.initializers.Zeros()))

    #loss=keras.losses.MAE()
    #opt = optimizers.Adam(learning_rate=0.01)
    # mse 对离群点权重赋值较高（需要求均值）
    # mae 对损失适用于训练数据被离群点损坏的时候（求中位数）
    #model.compile(optimizer=par['optimizer'], loss="mean_squared_logarithmic_error", metrics=['acc','mae'])
    # callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1)#使用loss作为监测数据，轮数设置为1
    x = tf.convert_to_tensor(train_X, tf.double, name='t')

    # 拟合很多个模型，直到拟合的模型为非增函数为止，选取前一个增函数（如果用tf自己写原生的梯度下降函数会快点）

    flag = 0
    i = 1
    step_flag=1
    while flag == 0:
        i = i+1
        if i % step_flag== 0:  # 每500次就检查一次这个DNN拟合的函数是否为增函数。，如果非增，就选择上一次检查过的增函数，并中断
            model.fit(train_X, train_Y, epochs=i, verbose=0)
            x = tf.convert_to_tensor(train_X, tf.double, name='t')

            with tf.GradientTape() as tape:
                tape.watch(x)
                y = model(x)
                dy1_dx = tape.gradient(target=y, sources=x)  # 求梯度,
                #print(dy1_dx)
                y2=np.array(y).copy()
                for m in range(0,len(y2)):
                    y2[m]=abs(y2[m]-train_Y[m])
                stds=y2.sum()
                # if stds>1000:
                #     learning_rate=0.01
                #     model.compile(optimizer=optimizer, loss=loss, metrics=['acc','mae'])
                # elif stds<1000 and stds>500:
                #     learning_rate=0.01
                #     model.compile(optimizer=optimizer, loss=loss, metrics=['acc','mae'])
                # elif  stds<500 and stds>300:
                #     learning_rate=0.001
                #     model.compile(optimizer=optimizer, loss=loss, metrics=['acc','mae'])
                # else:
                #     learning_rate=0.0001
                #     model.compile(optimizer=optimizer, loss=loss, metrics=['acc','mae'])
                print(y2.sum())
                score=r2_score(y,train_Y)
                print('score',score)
                if score>0:
                    step_flag=1
              
                if y2.sum()<0.00000000001:
                    break
                #if np.mean(dy1_dx) == 0:  # 防止梯度消失
                    #flag = 1
                if i % 50== 0:
                    fig666 = plt.figure()
                    # # 新建子图1
                    ax1 = fig666.add_subplot()
                    f1 = ax1.scatter(train_X, train_Y)

                    f2 = ax1.plot(x, y, c="r")
                    ax1.legend([f1, f2[0]], ['points', 'DNN'])
                    plt.savefig('img\\'+str(i)+".png")

                for i2 in dy1_dx:
                    tt = i2.numpy()[0]
                    if tt < -0.01:
                        flag = 1
                        break
                if score>0.9999:
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


        fig666 = plt.figure()
                # 新建子图1
        ax1 = fig666.add_subplot()
        f1 = ax1.scatter(train_X, train_Y,c='royalblue')

        f2 = ax1.plot(x, y, c="r")
        ax1.legend([f1, f2[0]], ['points', 'DNN model'])
        plt.savefig("img\\DNN-seg.png")
   

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
