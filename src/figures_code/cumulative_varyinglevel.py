# This code is for producing figure 2

import numpy as np
import matplotlib.pyplot as plt  
from matplotlib.ticker import FuncFormatter
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


x1=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
y1=[0.99,0.99,0.97,0.97,0.97,0.97,0.96,0.94,0.94,0.93,0.92,0.90,0.88,0.84,0.79,0.76,0.75,0.71,0.68,0.64,0.58]
x2=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
y2=[0.99,0.99,0.96,0.90,0.85,0.81,0.79,0.78,0.72,0.70,0.64,0.56,0.46,0.20,0.01,0,0,0,0,0,0]

x=np.arange(0,1)
l1=plt.plot(x1,y1,'xkcd:orange',label='Auto-Attack',marker='^')
l2=plt.plot(x2,y2,'xkcd:sky blue',label='GREAT Score',marker='s')
plt.grid(linestyle='-.')
plt.plot(x1,y1,'xkcd:orange',x2,y2,'xkcd:sky blue')
def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.xlabel('Perturbation Level',fontsize=14)
plt.ylabel('Robust Accuracy',fontsize=14)
plt.legend()
plt.show()
plt.savefig('/ground/pictures/figure_2_3.png')