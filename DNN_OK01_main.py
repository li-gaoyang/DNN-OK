# -*- coding: UTF-8 -*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

import DNN_OK02_semi
from tensorflow.keras.models import load_model
import matplotlib.image as mpimg  # mpimg 用于读取图片
import pandas as pd
import scipy.interpolate as si
import publicHelper as ph
import threadpool
import time
import normal_test
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

z_map1 = []
z_map2 = []


re_vals = []
  

def get_return_val(request, v):
    global re_vals
    re_vals.append(v)



if __name__ == "__main__":

    # 读取图片
    dem = cv.imread('DEM01.jpg')  # 读取原始图片
    dem = cv.cvtColor(dem, cv.COLOR_BGR2GRAY)  # 原始图片转灰度图
    dem = cv.resize(dem, (100, 100))  # 原始图片设定为100*100像素的
 
    # 选取100个样本数据，插值用（从原始图片里面找）
    # dem=dst[0:100, 0:100]#截取100*100的图片
    LEN_AREA = 100.0  # area length [m]#工区长度100*100的图片，所以工区长度为100
    # plt.imshow(dem, cmap = plt.get_cmap("gray")) # 显示图片
    N_DIV = 11  # 等分取样，11表示取(100/10)^2=100个样本
    xyzs = []
    n = np.linspace(0, LEN_AREA, N_DIV).astype(int)[:10]  # 取点 float转int 取前10个数
    x = []  # 存放样本点的x坐标
    y = []  # 存放样本点的y坐标
    z = []  # 存放样本点的z坐标
    for i in range(0, len(n)):
        for j in range(0, len(n)):
            xyz = [n[i], n[j], dem[n[i]][n[j]]]
            x.append(n[i])
            y.append(n[j]) 
            z.append(dem[n[i]][n[j]])
            xyzs.append(xyz)
    #样本正态分布检验
    abnormal=normal_test.Normal_Test(np.array(z,dtype=np.float))
    print(abnormal)
    # xyzs2=[]
    # for i in range(0,len(xyzs)):
    #     if i in abnormal_val_idx:
    #         print("非正态异常值",str(i))
    #     else:
    #         xyzs2.append(xyzs[i])
    # xyzs=xyzs2
    # for i in range(0,len(abnormal_val_idx)):
    #     z.pop(i)
    # 100个点创建距离协方差矩阵（通过数据处理，使协方差矩阵符合正态分布）
    DCOR = 100.0  # correlation distance [m] 相关距离
    STDEV = 8.0  # standard deviation 标准差
    # gen variance-covariance matrix 创建变异数协方差矩阵
    cov = ph.genCovMat(n, n, DCOR, STDEV)
    xyzs_np = np.array(xyzs)

    # 在100个点里面均匀取值选取20个样本点用来拟合半变异函数
    D_MAX = 100.0  # maximum distance in semivariogram modeling 半变异函数建模中的最大距离（选取的20个点在某一范围内）
    N_SEMIVAR = 20  # number of points for averaging empirical semivariograms 平均经验半变异函数的点数
    d_sv, sv = ph.genSemivar(xyzs_np, D_MAX, N_SEMIVAR, len(n)*len(n))  # 根据原始数据，D_MAX最大距离，N_SEMIVAR个数限制，
    
    # plt.scatter(d_sv, sv)  # 20个三点
    print(d_sv)
    print(sv)
    d_sv = np.insert(d_sv, 0, 0)
    sv = np.insert(sv, 0, 0)#快金为0
    model=DNN_OK02_semi.trains(d_sv,sv)  #DNN拟合半变异函数

    model = load_model("model_DNN.h5")
    #显示拟合的半变异函数
    # S1=plt.scatter(d_sv, sv,c='royalblue')  # 18个三点
    # S2=plt.plot(d_sv, model.predict(d_sv),c='red')  # 神经网络拟合的曲线
    # plt.legend([S1, S2[0]], ['Points', 'DNN Model'])
    # # plt.savefig("semi.jpg")
    # plt.show()
    #创建一个空白图片50*50的。
    N_DIV = 51
    dis_num=N_DIV/4
    #dis_num=30
    x_valid = np.linspace(0, LEN_AREA, N_DIV).astype(
        int)[:N_DIV-1]  # 工区是50*50，预测x范围是0到50共50个数
    y_valid = np.linspace(0, LEN_AREA, N_DIV).astype(
        int)[:N_DIV-1]  # 工区是50*50，预测y范围是0到50共50个数
    z_map1 = np.zeros([len(x_valid), len(y_valid)])
    z_map2 = np.zeros([len(x_valid), len(y_valid)])

    z_map0 = np.zeros([len(x_valid), len(y_valid)])  # 原图
    for i in range(len(x_valid)):
        for j in range(len(y_valid)):
            z_map0[j][i] = dem[x_valid[i]][y_valid[j]]

    # cv.imencode('.jpg', z_map0)[1].tofile('50X50原图.jpg')
    #存放克里金方差
    z_map1_error = np.zeros([len(x_valid-1), len(y_valid-1)])

    error_threshold = 20
    
    tempnps=[]
    # 神经网络克里金（单线程）
    for i in range(0,len(y_valid)):
        for j in range(len(x_valid)):
            # print("原始",i,j,z_map0[i][j])
      
            v, dnn_ok_error,tempnp = ph.dnn_ok_v(
                [x, y, z, x_valid[j], y_valid[i], model, error_threshold, i, j, 10])
            z_map1_error[i][j] =abs(dnn_ok_error)

            if v==-999:
                z_map1[i][j] = z_map1[i][j-1]
            else:
                z_map1[i][j] = v
            tempnps.append(tempnp)
            if v>300:
                print(v)
            if v<0:
                print(v)
            print("dnn_v:", v, "-------dnn_ok_error:",
                  dnn_ok_error, "-------", i, "--------", j)

            temp = 0


    tempnps=np.array(tempnps)
    with open('DNN_P.txt', 'w') as f:
        for i in range(len(tempnps)):
            for j in tempnps[i]:

                f.write(str(j)+' ')
            f.write('\n')
   

    # ph.savenp("z_map1", z_map1)
   

    # ph.savenp("1_0", z_map1-z_map0)
    # ph.savenp("2_0", z_map2-z_map0)
    # # 保存容差图片

    # ph.savenp("z_map1_error", z_map1_error)

    fig1 = plt.figure()
    fig1.canvas.set_window_title('原图')
    plt.imshow(z_map0, cmap="hsv", vmin=z_map0.min()
, vmax=z_map0.max())
    plt.colorbar()
    #plt.savefig("DEM01原图.jpg")
    fig2 = plt.figure()
    fig2.canvas.set_window_title('DNN_OK')
    plt.imshow(z_map1, cmap="hsv", vmin=z_map0.min(), vmax=z_map0.max())
    plt.colorbar()
    #plt.savefig("DNN_OK.jpg")
    # fig3 = plt.figure()
    # fig3.canvas.set_window_title('DNN_OK_error')
    # plt.imshow(z_map1_error, cmap="hsv", vmin=0, vmax=200)
    # plt.colorbar()
    #plt.savefig("DNN_OK_error.jpg")
    
    # print("DNN插值结果减原图的标准差:",np.std(z_map1-z_map0))
    print("DNN_RMSE = ",str(np.sqrt(mean_squared_error(z_map1,z_map0))))#均方根误差RMSE
    print("DNN_MAE = ", str(mean_absolute_error(z_map1,z_map0)))#平均绝对误差MAE
    # print("DNN_r2_score = ", str(r2_score(z_map1,z_map0)))# 
    testdata=model.predict(d_sv)
    print("DNN变差函数拟合_r2_score = ", str(r2_score(testdata,sv)))# 
    
    # print("DNN变差函数拟合_std = ", str(np.std(testdata-sv)))# 
    print("DNN插值结果的克里金方差的平均数:",abs(np.sum(z_map1_error)/z_map1_error.size))
    plt.show()
   