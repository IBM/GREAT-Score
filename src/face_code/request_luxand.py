'''
This is the api for luxand 
'''

#!/usr/bin/env python3
import requests
import os
import numpy as np

url = "https://api.luxand.cloud/photo/detect"

import time 


h=1 # 0 to 3


attribute_name_list=['old','young','with-eyeglasses','without-eyeglasses','overall']
attribute_name=attribute_name_list[h]

load_file_directory=['/interfacegan/results/age-conditioned-old-balanced','/interfacegan/results/stylegan_celebahq_gender_editing_young-balanced','/interfacegan/results/with-eyeglasses1-balanced','/interfacegan/results/without-eyeglasses-balanced']

load_img_name=load_file_directory[h]

url_name=attribute_name+'.txt'

f=open(url_name)



online_api_name='luxand'

store_list=[]
dir = load_img_name
imgList = os.listdir(dir)
#print(imgList)
imgList.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))
payload = {}
headers = { 'token': "" } # input your own token

#print(imgList)
#for count in range(0, len(imgList)):
for count in range(0,500):
    im_name = imgList[count]
    im_path = os.path.join(dir,im_name)
    print(im_path)
    files = { "photo": open(im_path, "rb") }
    response = requests.request("POST", url, data=payload, headers=headers, files=files)


    if response.status_code == 200:
        store_list.append(response.json())
        print (response.json())
    else:
        time.sleep(5)
        response = requests.request("POST", url, data=payload, headers=headers, files=files)
        if response.status_code == 200:
            store_list.append(response.json())
            print(response.json())
        else:
            store_list.append(0)
            print(114514)
   

    #store_list.append(response)

a=np.array(store_list)

load_file_name=online_api_name+'api'+attribute_name+'.npy'

np.save(load_file_name,a)


