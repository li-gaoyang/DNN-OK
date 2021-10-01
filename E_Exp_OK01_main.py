# -*- coding: UTF-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

import matplotlib.image as mpimg # mpimg 用于读取图片

import scipy.interpolate as si
import publicHelper_exp as ph
import threadpool
import time
z_map1=[]
z_map2=[]

import cv2 as cv



re_vals=[]

def get_return_val(request,v):
    global re_vals
    re_vals.append(v)







if __name__=="__main__":


    dem=cv.imread('DEM01.jpg')#读取原始图片
    dem=cv.cvtColor(dem,cv.COLOR_BGR2GRAY) #原始图片转灰度图
    dem = cv.resize(dem, (100,100)) #原始图片设定为100*100像素的
    # dem=dst[0:100, 0:100]#截取100*100的图片
    LEN_AREA = 100.0 #area length [m]#工区长度100*100的图片，所以工区长度为100
    # plt.imshow(dem, cmap = plt.get_cmap("gray")) # 显示图片
    N_DIV=11#等分取样，11表示取(100/10)^2=100个样本
    xyzs=[]
    n = np.linspace(0, LEN_AREA, N_DIV).astype(int)[:10]#取点 float转int 取前9个数
 
    x=[]#存放样本点的x坐标
    y=[]#存放样本点的y坐标
    z=[]#存放样本点的z坐标
    for i in range(0,len(n)): 
        for j in range(0,len(n)):
            xyz=[n[i],n[j],dem[n[i]][n[j]]]
            x.append(n[i])
            y.append(n[j])
            z.append(dem[n[i]][n[j]])
            xyzs.append(xyz)
        

    
    DCOR = 100.0 #correlation distance [m] 相关距离,
    STDEV = 8.0 #standard deviation 标准差
     
    cov = ph.genCovMat(n, n, DCOR, STDEV) #gen variance-covariance matrix 创建变异数协方差矩阵
    xyzs_np=np.array(xyzs)
    D_MAX = 100.0 #maximum distance in semivariogram modeling 半变异函数建模中的最大距离
    N_SEMIVAR = 20 #number of points for averaging empirical semivariograms 平均经验半变异函数的点数
    d_sv, sv = ph.genSemivar(xyzs_np, D_MAX, N_SEMIVAR,len(n)*len(n))#根据原始数据，D_MAX最大距离，N_SEMIVAR个数限制，
    d_sv = np.insert(d_sv, 0, 0)
    sv = np.insert(sv, 0, 0)#快金为0
    
  
    plt.scatter(d_sv,sv)#20个三点
  

   
 
   
    param = ph.semivarFitting(d_sv, sv)#vb
    print(param)
    S1=plt.scatter(d_sv,sv,c='royalblue')#20个三点
 
   
    # param[0]=0#快金为0
    
    S2=plt.plot(d_sv,ph.semivar_exp2(d_sv, param[0], param[1], param[2]),c='red')#指数模型拟合的曲线
    # plt.savefig("chart.jpg")
    #plt.show()
    plt.legend([S1, S2[0]], ['Points', 'Exponential Model'])
    # plt.show()
    '''plot empirical/      semivariogram 情节经验/理论变异函数'''
    # d_fit = np.linspace(0.0, D_MAX, 1000)#生成1000个等间隔的数字（在0到500之间）
 

    '''Ordinary Kriging'''
   
    N_DIV = 51
    dis_num=N_DIV/4
    x_valid = np.linspace(0, LEN_AREA, N_DIV).astype(int)[:N_DIV-1]#工区是50*50，预测x范围是0到50共50个数
    y_valid = np.linspace(0, LEN_AREA, N_DIV).astype(int)[:N_DIV-1]#工区是50*50，预测y范围是0到50共50个数
    z_map1 = np.zeros([len(x_valid), len(y_valid)])
    z_map2 = np.zeros([len(x_valid), len(y_valid)])

    z_map0 = np.zeros([len(x_valid), len(y_valid)])#原图
    for i in range(len(x_valid)):
        for j in range(len(y_valid)):
            z_map0[j][i]=dem[x_valid[i]][y_valid[j]]
    
    cv.imencode('.jpg', z_map0)[1].tofile('z_map0原图.jpg')

 
    z_map1_error = np.zeros([len(x_valid-1), len(y_valid-1)])
    z_map2_error = np.zeros([len(x_valid-1), len(y_valid-1)])
    
   
    error_threshold=20
    #普通克里金
    for i in range(len(y_valid)):
        for j in range(len(x_valid)):
            est2,error2=ph. mat_ok2(x, y, z,x_valid[j],y_valid[i],param,error_threshold)#OK
            z_map2_error[i][j]=abs(error2)
            z_map2[i][j]=est2
  
 
    ph.savenp("z_map1", z_map1)
    ph.savenp("z_map1", z_map2)
  
    ph.savenp("1_0",z_map1-z_map0)
    ph.savenp("2_0",z_map2-z_map0)
    #保存容差图片
  
    ph.savenp("z_map1_error", z_map1_error) 
    ph.savenp("z_map2_error", z_map2_error)


    fig3=plt.figure()
    fig3.canvas.set_window_title('EXP_OK')
    plt.imshow(z_map2, cmap = "hsv",vmin=z_map0.min(), vmax=z_map0.max())
    plt.colorbar()
    

    fig4=plt.figure()
    fig4.canvas.set_window_title('原图')
    plt.imshow(z_map0, cmap = "hsv",vmin=z_map0.min(), vmax=z_map0.max())
    plt.colorbar()
    

    # fig5=plt.figure()
    # fig5.canvas.set_window_title('EXP_OK_原图_err')
    # plt.imshow(z_map2-z_map0, cmap = "rainbow",vmin=-60, vmax=45)
    # plt.colorbar()
  
    

    fig1=plt.figure()
    fig1.canvas.set_window_title('EXP_kriging_error')
    plt.imshow(z_map2_error, cmap = "hsv",vmin=0, vmax=280)
    plt.colorbar()
  
    print("指数模型插值结果-原图的标准差:",np.std(z_map2-z_map0))


    print("指数模型克里金方差的平均数:",np.sum(z_map2_error)/z_map2_error.size)

 
    z_map1_er=0
    z_map2_er=0
    for i in range(len(z_map1)):
        for j in range(len(z_map1)):
            e1=z_map1[i][j]-z_map0[i][j]
            e2=z_map2[i][j]-z_map0[i][j]
            if e1<e2:
                z_map1_er=z_map1_er+1
            if e1>e2:
                z_map2_er=z_map2_er+1

 
    print("z_map2_er:",z_map2_er)

    plt.show()





