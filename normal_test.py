from re import U
import numpy as np
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
from scipy.stats import norm
from scipy.stats import kstest




#正态分布检验
def Normal_Test(a1):
    
    #abnormal_value(a1)
    u = a1.mean()  # 计算均值
    std = a1.std()  # 计算标准差
    abnormal_val_idx=[]

    
    # a2=[]
    # for i in range(0,len(a1)):
    #     if abs(a1[i]-u)>3*std:
      
    #         abnormal_val_idx.append(i)
    #     else:
    #         a2.append(a1[i])

     
    s = pd.DataFrame(a1,columns = ['P-value'])
    p_value1=kstest(s['P-value'], 'norm', (u, std))[1]

    print("p_value1",p_value1)
  
   
    

    # fig111,ax2 = plt.subplots()
    # fig111.canvas.set_window_title('正态分布')
  

    # s.hist(bins=30,alpha = 0.5,ax = ax2)
    # line=s.plot(kind = 'kde', secondary_y=True,ax = ax2) 
    # ax2.set_title("P-value="+str(round(p_value1,2)), fontsize=24)
    # ax2.set_xlabel("", fontsize=24)
    # ax2.set_ylabel("", fontsize=24)
    # plt.grid()
    # #plt.savefig("Normal_test.png")
    
    # plt.show()
    # 绘制直方图
    # 呈现较明显的正太性
    print("p_value值",p_value1)
    if p_value1>=0.05:
        return "符合正态分布"
    else:
        return "不符合正态分布" 

    plt.show()
    

if __name__=="__main__":
    a=[218., 211., 223., 227., 194., 200., 228., 220., 209., 190., 200.,
       194., 199., 209., 206., 196., 194., 194., 201., 202., 191., 224.,
       185., 212., 205., 192., 220., 207., 157., 205., 237., 205., 208.,
       220., 195., 181., 212., 213., 153., 195., 194., 199., 188., 204.,
       208., 184., 132., 161., 139., 163., 223., 216., 216., 217., 194.,
       224., 144., 135., 112., 111., 197., 255., 246., 216., 241., 217.,
       165., 167., 139.,  86., 193., 197., 217., 237., 217., 214., 189.,
       163., 119., 113., 215., 199., 194., 240., 236., 222., 177., 226.,
       167.,  98., 176., 215., 233., 224., 219., 221., 212., 227., 193.,
       125.]
    aa=np.array(a,dtype=float)
    Normal_Test(aa)