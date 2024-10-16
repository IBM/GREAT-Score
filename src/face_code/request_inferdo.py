'''
https://rapidapi.com/inferdo/api/face-detection6

tips: when you finish the code in the for loop, remeber to tab!
'''



import requests
import os 
import requests
import numpy as np

import time 


online_api_name_list=['betaface','inferdo','arsa-technology','deepface','baidu','microsoft']
online_api_name='inferdo'


attribute_name_list=['old','young','with-eyeglasses','without-eyeglasses','overall']

h=2 # 0 to 3

attribute_name=attribute_name_list[h]

url_name=attribute_name+'.txt'

f=open(url_name)


line=f.readline()

url = "https://face-detection6.p.rapidapi.com/img/face-age-gender"

headers = {
	"content-type": "application/json",
	"X-RapidAPI-Key": "",# input your own api key
	"X-RapidAPI-Host": "face-detection6.p.rapidapi.com"
}
store_list=[]

while line:
	
	url1=line   
	print(url1)
	payload = {
		"url": url1,
		"accuracy_boost": 4
	}

	response = requests.request("POST", url, json=payload, headers=headers)
	if response.status_code == 200:
		store_list.append(response.json())
		print(response.json())
	else:
		time.sleep(5)
		response = requests.request("POST", url, json=payload, headers=headers)
		if response.status_code == 200:
			store_list.append(response.json())
			print(response.json())
		else:
			store_list.append(0)
			print(114514)
	
	line=f.readline()


'''
we can use this to get the probability of each image gender
'''

a=np.array(store_list)

load_file_name=online_api_name+'api'+attribute_name+'.npy'

np.save(load_file_name,a)

f.close()