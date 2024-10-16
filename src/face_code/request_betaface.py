'''
https://rapidapi.com/betaface/api/face-recognition
'''

# url_name refers to the file you store the urls for images

import numpy as np
import os 
import requests
import time
online_api_name_list=['betaface','inferdo','arsa-technology','deepface','baidu','microsoft']
online_api_name='betaface'

attribute_name_list=['old','young','with-eyeglasses','without-eyeglasses','overall']

h=3 # 0 to 3

attribute_name=attribute_name_list[h]

url_name=attribute_name+'.txt'

f=open(url_name)
line=f.readline()
url = "https://betaface-face-recognition-v1.p.rapidapi.com/media"
headers = {
	"content-type": "application/json",
	"X-RapidAPI-Key": "",# input your own api key
	"X-RapidAPI-Host": "betaface-face-recognition-v1.p.rapidapi.com"
}

store_list=[]


while line:
	
	url1=line
	print(url1)   
	payload = {
	"file_uri": url1,
	"detection_flags": "propoints,classifiers,content",
	"recognize_targets": ["all@celebrities.betaface.com"],
	"original_filename": "sample.png"
}
	#print(line)
	#print(type(line))
	response = requests.request("POST", url, json=payload, headers=headers)
	if response.status_code == 200:
		store_list.append(response.json())
		print(response)
	else:
		store_list.append(0)
		print(114514)
		

	
	line=f.readline()


a=np.array(store_list)

load_file_name=online_api_name+'api'+attribute_name+'.npy'

np.save(load_file_name,a)
'''
we can detect 0 attribute here , also , we need to write different name of the npy file to store different online api results, maybe the name?
'''




f.close()