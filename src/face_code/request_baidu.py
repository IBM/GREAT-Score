

import os 
import urllib3
from urllib.parse import urlencode
import urllib
import urllib.request
import sys
import ssl
import base64
import json
import numpy as np
import time
'''

f = open(imagePath, 'rb')

img = base64.b64encode(f.read())
img = img.decode('utf-8')
'''


'''

This is for baidu api.
'''
# encoding:utf-8

import requests

'''

'''

request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
import os 
import requests


access_token = '' # input your own access_token
headers = {'content-type': 'application/json'}



request_url = request_url + "?access_token=" + access_token

h=3 # 0 to 3


attribute_name_list=['old','young','with-eyeglasses','without-eyeglasses','overall']
attribute_name=attribute_name_list[h]

load_file_directory=['/interfacegan/results/age-conditioned-old-balanced','/interfacegan/results/stylegan_celebahq_gender_editing_young-balanced','/interfacegan/results/with-eyeglasses1-balanced','/interfacegan/results/without-eyeglasses-balanced']

load_img_name=load_file_directory[h]

url_name=attribute_name+'.txt'

f=open(url_name)

online_api_name='baidu'

store_list=[]
dir = load_img_name
imgList = os.listdir(dir)
#print(imgList)
imgList.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))

#print(imgList)
#for count in range(0, len(imgList)):
for count in range(0,500):
    im_name = imgList[count]
    im_path = os.path.join(dir,im_name)
    print(im_path)
    imagePath=im_path
    f = open(imagePath, 'rb')
    img = base64.b64encode(f.read())
    img = img.decode('utf-8')
    params = {"image": img,"image_type":"BASE64","face_field":"age,beauty,glasses,gender,race"}
    response = requests.post(request_url, data=params, headers=headers)
    if response.json()['error_code']==0:
        print (response.json())
    else:
        time.sleep(5)
        response = requests.post(request_url, data=params, headers=headers)
        print (response.json())
   
    store_list.append(response.json())

a=np.array(store_list)

load_file_name=online_api_name+'api'+attribute_name+'.npy'

np.save(load_file_name,a)


