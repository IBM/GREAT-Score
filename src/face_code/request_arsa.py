'''
https://rapidapi.com/arsa-technology-arsa-technology-default/api/face-recognition18/
'''




import os 
import requests
import numpy as np
import time


online_api_name_list=['betaface','inferdo','arsa-technology','deepface','baidu','microsoft']
online_api_name='arsa-technology'


attribute_name_list=['old','young','with-eyeglasses','without-eyeglasses','overall']

h=3 # 0 to 3

attribute_name=attribute_name_list[h]

url_name=attribute_name+'.txt'

f=open(url_name)




line=f.readline()

headers = {
	"content-type": "application/json",
	"X-RapidAPI-Key": "",  # input your own api key
	"X-RapidAPI-Host": "face-recognition18.p.rapidapi.com"
}
url = "https://face-recognition18.p.rapidapi.com/recognize_face2"

store_list=[]

while line:
	
	url1=line   
	payload = {"image_input_url": url1}
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


a=np.array(store_list)

load_file_name=online_api_name+'api'+attribute_name+'.npy'

np.save(load_file_name,a)

'''
We also need to define the gender label
'''
