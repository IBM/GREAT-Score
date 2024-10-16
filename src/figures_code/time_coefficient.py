# This code is for producing Figure4.


import matplotlib.pyplot as plt
import numpy as np
time=np.load("time.npy")  # Save the time of each model's time consuming.

autoattack_time=np.array([30436,29793, 30668.1,5191.4, 5048 , 5183.1, 5052.8,2205.7,2232.2,30373, 1932.6,5491.3,3457.4,3323.1,1788.8,4291.7,1809.7])

#y=[1670.5,2005.5,904,1614.5,1185.965833]

#y=autoattack_time/time


#label_list = ['F', 'U', 'R','P','O']    # 横坐标刻度显示值
xx = range(17)
plt.figure().set_size_inches(10,6)
rects1 = plt.bar(x=xx, height=y, width=0.4, alpha=0.8, color='blue',align='center')
     # y轴取值范围
plt.ylabel("Run-time improvement ratio",color='blue',fontsize=20)
plt.xlabel("Model Index",color="green",fontsize=20)
plt.xticks(np.arange(0, 17, 1))
#plt.xlabel("Group Name")
plt.savefig("Time Coefficient.png")
plt.show()
