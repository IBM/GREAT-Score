# This code is for producing Figure 1 in the paper
 
import matplotlib
import pandas as pd
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.pyplot import MultipleLocator
import random


index=[]
for x in range(20):
   a=random.randint(0,500)
   index.append(a)
index=np.array(index)
Image_Id = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
difference_file_name='Rebuffi2021Fixing_70_16_cutmix_extra'+'difference_result'+'.npy'   # Great score file
path1='Rebuffi2021Fixing_70_16_cutmix_extra'+'_final'+'.npy' # cw attack file

fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
l2_norm = np.load(path1)[index]
l2_norm1=np.load(difference_file_name)[index]
 
data_plot = pd.DataFrame({"Image Id":Image_Id , "L2 Perturbation Level":l2_norm,"L2 Perturbation Level1":l2_norm1})
plt.figure().set_size_inches(10,6)
gfg1=sns.scatterplot(x = "Image Id", y = "L2 Perturbation Level", data=data_plot,label='CW Attack',marker='^',s=100)
gfg1.legend(fontsize=14)
gfg2=sns.scatterplot(x = "Image Id", y = "L2 Perturbation Level1" ,data=data_plot,label='GREAT Score(local)',s=100)
gfg2.legend(fontsize=14)
x_major_locator=MultipleLocator(1)

plt.xlabel('Image ID',fontsize=20)
plt.ylabel("L2 Perturbation Level",fontsize=20)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()

plt.savefig('/ground/pictures/figure_2_2.png')