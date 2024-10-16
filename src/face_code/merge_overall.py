'''
We assume the input label is a npz file with (40,500,1) shape, we may need other functions to tell the labels, assume the name is y_npz
'''
'''
import numpy as np 

'''

import os
import numpy as np
import math

from collections import defaultdict
#list
d = defaultdict(list)


'''
From slim-cnn, it's locally running models instead of online api
'''
number_imgs=500
online_api_name_list=['betaface','inferdo','arsa-technology','deepface','baidu','luxand','silm-cnn','FAN']

'''
the conda env is caffe-gpu for FAN
'''
online_api_name="luxand"
attribute_name1="Gender"

attribute_name_list=['old','young','with-eyeglasses','without-eyeglasses','overall']

h=2 # 0 to 3

attribute_name=attribute_name_list[h]

load_file_name=online_api_name+'api'+attribute_name+'.npy'


ground_truth_name=attribute_name+'.npy'

load_file=np.load(load_file_name,allow_pickle=True)
y_npz=np.load(ground_truth_name,allow_pickle=True)
data=open("final_online_api_score.txt",'a')



for i in range (0,1):
    online_api_name=online_api_name_list[i]
    result=[]
    for j in range(4):
        attribute_name=attribute_name_list[j]
        load_file_name=online_api_name+'great'+attribute_name+'.npy'
        load_file=np.load(load_file_name,allow_pickle=True)
        result.append(load_file)
    print(online_api_name)
    print((np.mean(result)))
   