
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import os
from tensorflow.keras.models import load_model
import time

import scipy.interpolate as si
''' sampling via multivariate normal distribution with no trend  多变量正态分布抽样，无趋势'''


def genMultivariate(cov):
    print("--genMultivariate()--")
    L = np.linalg.cholesky(cov)
    z = np.random.standard_normal(len(cov))
    return np.dot(L, z)


'''gen variance-covariance matrix 得到协方差矩阵'''


def genCovMat(x, y, dcor, sigma):
    print("--genCovMat()--")
    dmat = distance(x, y, x[:, np.newaxis],
                    y[:, np.newaxis])  # distance matrix

    return dmat
    tmp = 0.6931471805599453 / dcor  # np.log(2.0)/dcor

    return sigma * sigma * np.exp(-dmat * tmp)  # 转正态分布？？？


'''measurement location 测量位置'''


def genMeasurementLocation(size, len_area):
    print("--genMeasurementLocation()--")
    x = np.random.uniform(0.0, len_area, size)
    y = np.random.uniform(0.0, len_area, size)
    return x, y


''' gen empirical semivariogram via binning  通过装箱生成经验半变异函数'''


# data表示[x,y,z]数据集，d_max表示搜索距离，num表示我要几个点（20个），N表示样本个数
def genSemivar(data, d_max, num, N):
    def genCombinations(arr):  # arr表示传入0到99共100个数
        r, c = np.triu_indices(len(arr), 1)  # 返回100*100的矩阵上三角，偏移为1
        return np.stack((arr[r], arr[c]), 1)

    print("--genSemivar()--")
    d_semivar = np.linspace(0.0, d_max, num)  # 0到100均匀取值总共取20个
    SPAN = d_semivar[1] - d_semivar[0]  # 等差数为5.26

    indx = genCombinations(np.arange(N))

    d = distance(data[indx[:, 0], 0], data[indx[:, 0], 1],
                 data[indx[:, 1], 0], data[indx[:, 1], 1])
    indx = indx[d <= d_max]
    d = d[d <= d_max]
    semivar = (data[indx[:, 0], 2] - data[indx[:, 1], 2])**2

    semivar_avg = np.empty(num)
    for i in range(num):
        d1 = d_semivar[i] - 0.5*SPAN
        d2 = d_semivar[i] + 0.5*SPAN
        indx_tmp = (d1 < d) * (d <= d2)  # index within calculation span
        semivar_tr = semivar[indx_tmp]
        semivar_avg[i] = semivar_tr.mean()

    d_sv = d_semivar[np.isnan(semivar_avg) == False]
    sv = 0.5 * semivar_avg[np.isnan(semivar_avg) == False]

    return d_sv, sv


'''theoretical semivariogram (exponential)  理论变异函数(指数)'''


def semivar_DNN(d, model):  # 传入距离h（d）,块金，基台，变程，带入指数模型进行计算

    max_d = 100
    if len(d.shape) == 1:
        if len(d)==0:
            d = np.array([0])
        res = model.predict(d)
        res2 = []
        for i in range(len(res)):
            # if res[i][0]>max_d:
            #     res[i][0]=max_d

            res2.append(res[i][0])

        return res2
    else:
        ls = []
        for i in range(len(d)):
            for j in range(len(d[i])):
                # if d[i][j]>max_d:
                #     d[i][j]=max_d
                ls.append(d[i][j])
        if len(ls)==0:
            ls.append(0)
        t = model.predict(ls)

        for i in range(len(d)):
            for j in range(len(d[i])):
                d[i][j] = t[i*len(d)+j]

        return d


'''theoretical semivariogram (exponential)  理论变异函数(指数)'''


def semivar_exp2(d, nug, sill, ran):  # 传入距离 h（d）,块金，基台，变程，带入指数模型进行计算

    t = np.abs(nug) + np.abs(sill) * (1.0-np.exp(-d/(np.abs(ran))))  # 指数模型
    # t=np.abs(nug) + np.abs(sill) * (1.0-np.exp(-np.square(d/(np.abs(ran)))))#高斯模型
    # t=np.abs(nug) + np.abs(sill) * (1.5*d/(np.abs(ran)-0.5*np.power(d/(np.abs(ran)),3)))
    # for i in range(0,len(d)):
    #     if d[i]>np.abs(ran):
    #         t[i]=np.abs(nug) + np.abs(sill)

    return t


'''fitting emperical semivariotram to theoretical model  获取块金，基台变程'''


def semivarFitting(d, data):  # 获取块金，基台，变程
    def objFunc(x):
        theorem = semivar_exp2(d, x[0], x[1], x[2])
        return ((data-theorem)**2).sum()

    x0 = np.random.uniform(0.0, 1.0, 3)
    res = minimize(objFunc, x0, method='nelder-mead')
    for i in range(5):
        x0 = np.random.uniform(0.0, 1.0, 3)
        res_tmp = minimize(objFunc, x0, method='nelder-mead')
        if res.fun > res_tmp.fun:
            res = res_tmp
    return np.abs(res.x)


# def drawimg():


# 神经网络克里金
def ordinaryKriging1(mat, x_vec, y_vec, z_vec, x_rx, y_rx, model):
    savenp("mat1", mat)
    vec = np.ones(len(z_vec)+1, dtype=np.float)
    d_vec = distance(x_vec, y_vec, x_rx, y_rx)
    if len(d_vec)==0:
        return -999,0
    vec[:len(z_vec)] = semivar_DNN(d_vec, model)
    savenp("vec1", vec)
    # savenp("inv_mat_befo1", mat)
    weight = np.linalg.solve(mat, vec)


    savenp("weight1", weight)
    est = (z_vec * weight[:len(z_vec)]).sum()
    est = 0

    savenp("weight1", weight)
    est = (z_vec * weight[:len(z_vec)]).sum()
    n0 = mat[0][0]
    n1 = (z_vec * weight[:len(z_vec)]).sum()
    n2 = weight[len(weight)-1]
    er = n0-n1-n2
    ok1_error = abs(er)
    # for i in range(len(weight)-1):
    #     est = weight[i]*z_vec[i]+est

    # ok1_error=0
    # for i in range(len(weight)-1):
    #     ok1_error = weight[i]*(z_vec[i]-est)*(z_vec[i]-est)/2+ok1_error
    # ok1_error=ok1_error/(len(weight)-1)

    return est, ok1_error

# 普通克里金


def ordinaryKriging2(mat, x_vec, y_vec, z_vec, x_rx, y_rx, nug, sill, ran):
    savenp("mat2", mat)
    vec = np.ones(len(z_vec)+1, dtype=np.float)
    d_vec = distance(x_vec, y_vec, x_rx, y_rx)
    vec[:len(z_vec)] = semivar_exp2(d_vec, nug, sill, ran)
    savenp("vec2", vec)

    weight = np.linalg.solve(mat, vec)
    savenp("weight2", weight)
    est = (z_vec * weight[:len(z_vec)]).sum()
    n0 = mat[0][0]
    n1 = (z_vec * weight[:len(z_vec)]).sum()
    n2 = weight[len(weight)-1]
    ok1_error = cal_error(n0, n1, n2)

    # print("ok2",ok1_error,est)
    return est, ok1_error


def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


'''matrix for weight calculation in Ordinary Kriging 普通克里格法中计算权重的矩阵'''


def genMat1(x_vec, y_vec, z_vec, x, y, model, dis_num):  # 神经网络克里金计算权重矩阵
    x_vec1 = np.array([], dtype="float32")
    y_vec1 = np.array([], dtype="float32")
    z_vec1 = np.array([], dtype="float32")
    for i in range(len(z_vec)):
        dis = distance(x_vec[i], y_vec[i], x, y)
        if dis < dis_num:  # 有问题？？？？？？？
            x_vec1 = np.append(x_vec1, x_vec[i])
            y_vec1 = np.append(y_vec1, y_vec[i])
            z_vec1 = np.append(z_vec1, z_vec[i])

    mat = distance(
        x_vec1, y_vec1, x_vec1[:, np.newaxis], y_vec1[:, np.newaxis])

    mat = semivar_DNN(mat, model)

    # savenp("genMat1",mat)
    mat = np.vstack((mat, np.ones(len(z_vec1))))
    mat = np.hstack((mat, np.ones([len(z_vec1)+1, 1])))
    mat[len(z_vec1)][len(z_vec1)] = 0.0

    return x_vec1, y_vec1, z_vec1, mat


'''matrix for weight calculation in Ordinary Kriging 普通克里格法中计算权重的矩阵'''


def genMat2(x_vec, y_vec, z_vec, nug, sill, ran, x, y, dis_num):  # 普通克里金计算权重矩阵
    x_vec1 = np.array([], dtype="float32")
    y_vec1 = np.array([], dtype="float32")
    z_vec1 = np.array([], dtype="float32")
    for i in range(len(z_vec)):
        dis = distance(x_vec[i], y_vec[i], x, y)
        if dis < dis_num:
            x_vec1 = np.append(x_vec1, x_vec[i])
            y_vec1 = np.append(y_vec1, y_vec[i])
            z_vec1 = np.append(z_vec1, z_vec[i])

    mat = distance(
        x_vec1, y_vec1, x_vec1[:, np.newaxis], y_vec1[:, np.newaxis])

    mat = semivar_exp2(mat, nug, sill, ran)
    # savenp("genMat2",mat)
    mat = np.vstack((mat, np.ones(len(z_vec1))))
    mat = np.hstack((mat, np.ones([len(z_vec1)+1, 1])))
    mat[len(z_vec1)][len(z_vec1)] = 0.0

    return x_vec1, y_vec1, z_vec1, mat

# def fmt(x, y):
#     z2 = np.take(si.interp2d(X, Y, z_map1)(x, y), 0)
#     z3 = np.take(si.interp2d(X, Y, z_map2)(x, y), 0)
#     return 'x={x:.5f}  y={y:.5f}  z2={z2:.5f} z3={z3:.5f}'.format(x=x, y=y, z2=z2, z3=z3)


# plt.gca().format_coord = fmt

# 保存np矩阵
def savenp(fliename, mat):
    with open(fliename+'.txt', 'w') as f:
        for i in range(len(mat)):
            f.write(str(mat[i])+'\n')


def NormMinandMax(npdarr, min=0, max=1):
    """"
    将数据npdarr 归一化到[min,max]区间的方法
    返回 副本
    """
    arr = npdarr.flatten()
    Ymax = np.max(arr)  # 计算最大值
    Ymin = np.min(arr)  # 计算最小值
    k = (max - min) / (Ymax - Ymin)
    last = min + k * (arr - Ymin)

    return last


def cal_error(old_val, n, u):
    er = old_val-n-u
    er = abs(er)
    return er


# 神经网络克里金
def dnn_ok_v(pars):
    x = pars[0]
    y = pars[1]
    z = pars[2]
    x_valid_j = pars[3]
    y_valid_i = pars[4]

    model = pars[5]
    error_threshold = pars[6]
    i = pars[7]
    j = pars[8]
    dis_num = pars[9]
    x1, y1, z1, mat1 = genMat1(
        x, y, z, x_valid_j, y_valid_i, model, dis_num)

    # savenp("mat1", mat1)
    est, ok1_error = ordinaryKriging1(
        mat1, x1, y1, z1, x_valid_j, y_valid_i, model)
    # if est>255:
    #     est=255
    # if est<0:
    #     est=83
    # if ok1_error>error_threshold:
    #     ok1_error=0
    # else:
    #     ok1_error=255
    tempnp=[]
    for i in range(0,len(x1)):
        tempnp.append(z1[i])
    tempnp.append(est)
    
   
    return (est, ok1_error,tempnp)

#普通克里金  
def mat_ok2(x, y, z,x_valid_j,y_valid_i,param,dis_num):
    x2,y2,z2,mat2 =  genMat2(x, y, z, param[0], param[1], param[2], x_valid_j, y_valid_i,dis_num)
    # savenp("mat1", mat2)
    est,ok2_error = ordinaryKriging2(mat2, x2, y2, z2, x_valid_j,y_valid_i, param[0], param[1], param[2]) 
    
    # if ok2_error>error_threshold:
    #     ok2_error=0
    # else:
    #     ok2_error=255
            
    return est,ok2_error