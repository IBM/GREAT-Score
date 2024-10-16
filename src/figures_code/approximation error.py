# This code is for producing approximation error figure


from cmath import exp
import matplotlib.pyplot as plt
import numpy as np
import math


x = np.arange(0.1,0.3,0.02)
y1=[]
y2=[]
y3=[]
for t in x:
    y_1 = 32*math.exp(1)*(math.log(2/0.05))/(t*t)
    y_2 = 32*math.exp(1)*(math.log(2/0.15))/(t*t)
    y_3 = 32*math.exp(1)*(math.log(2/0.25))/(t*t)
    y1.append(y_1)
    y2.append(y_2)
    y3.append(y_3)

plt.xlabel("Approximation Error",fontsize=15)
plt.ylabel("Sample Complexity",fontsize=15)
plt.plot(x, y1,marker='o',label='$\delta$=0.05')
plt.plot(x, y2,marker='o',label='$\delta$=0.15')
plt.plot(x, y3,marker='o',label='$\delta$=0.25')

plt.legend()
plt.savefig("sample complexity.png")
plt.show()