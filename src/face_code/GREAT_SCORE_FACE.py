# since we can load images with url iteratively, now we want to read the labels by this file

'''
We assume the input label is a npz file with (40,500,1) shape, we may need other functions to tell the labels, assume the name is y_npz
'''
'''
import numpy as np 

'''

import os
import numpy as np
import math

'''
From slim-cnn, it's locally running models instead of online api
'''


'''
For deepface api, due to the ip address limit and citation , please go to there website to get the result
'''
number_imgs=500
online_api_name_list=['betaface','inferdo','arsa-technology','deepface','baidu','luxand','silm-cnn','FAN']

'''
the conda env is caffe-gpu for FAN
'''
online_api_name="betaface"
attribute_name1="Gender"

attribute_name_list=['old','young','with-eyeglasses','without-eyeglasses','overall']

h=3 # 0 to 3

attribute_name=attribute_name_list[h]

load_file_name=online_api_name+'api'+attribute_name+'.npy'

ground_truth_name=attribute_name+'.npy'

load_file=np.load(load_file_name,allow_pickle=True)
y_npz=np.load(ground_truth_name,allow_pickle=True)
data=open("final_online_api_score.txt",'a')
#print(load_file[1]['media']['faces'][0]['tags'][0]['confidence'])
'''
For Gender, we put ['Female','Male'] Order
'''



result1=[]
difference=np.zeros(500,dtype=float)

for i in range(number_imgs):
	label=y_npz[i]
	prediction_result=load_file[i]
	if online_api_name=="betaface" :
		response=load_file[i]
		if response==0 or response['media']['faces']==None:
			#print(1)
			logits_score=0.5
		else:
			#print(response['media']['faces'][0]['tags'][18]['value'])
			gender_label=response['media']['faces'][0]['tags'][18]['value']
			if gender_label=='female':
						logits_score=response['media']['faces'][0]['tags'][18]['confidence']
			else:
						logits_score=1-response['media']['faces'][0]['tags'][18]['confidence']
			
		
	elif online_api_name=='inferdo':
		if attribute_name1=="Gender":
			#for different name we use different index
				response=load_file[i]
				if response['detected_faces']==[]:
					logits_score=0.5
				else:
					gender_label=response['detected_faces'][0]['Gender']['Gender']
					if gender_label=='female':
						logits_score=response['detected_faces'][0]['Gender']['Probability']/100
					else:
						logits_score=1-response['detected_faces'][0]['Gender']['Probability']/100
	elif online_api_name=='arsa-technology':
		if attribute_name1=="Gender":
			    
			#for different name we use different index
				response=load_file[i]
				if response==0:
					logits_score=0.5
				else:
					gender_label=response['recognition_result'][0]['gender']
					if gender_label=='female':
						logits_score=response['recognition_result'][0]['gender_probability']
					else:
						logits_score=1-response['recognition_result'][0]['gender_probability']
	elif online_api_name=='deepface':
		if attribute_name1=="Gender":
			#for different name we use different index
				response=load_file[i]
				logits_score=response['gender']['Woman']/100
	elif online_api_name=='baidu':
		if attribute_name1=="Gender":
			#for different name we use different index
			    
				response=load_file[i]
				if response['error_code']==222202:
						logits_score=0.5
				else:
						gender_label=response['result']['face_list'][0]['gender']['type']
						if gender_label=='female':
								logits_score=response['result']['face_list'][0]['gender']['probability']
						else:
								logits_score=1-response['result']['face_list'][0]['gender']['probability']
	elif online_api_name=='luxand':
		if attribute_name1=="Gender":
			#for different name we use different index
				response=load_file[i]
				if response==[]:
					logits_score=0.5
				else:
					gender_label=response[0]['gender']['value']
					if gender_label=='Female':
							logits_score=response[0]['gender']['probability']
					else:
							logits_score=1-response[0]['gender']['probability']
	elif online_api_name=='slim-cnn':
		attribute_index=2 # this one can vary from 0 to 40.
		for i in range(3):
			response=load_file[i]
			logits_score=response[0][attribute_index]
	elif online_api_name=='FAN':
		attribute_index=2
		for i in range(3):
			response=load_file[i]
			print(response[attribute_index])


	
	#suppose we have got the score with shape 1 after the steps
	

	logits_list=[logits_score,1-logits_score]
	

	if logits_list[label]<= 0.5:
		difference[i]=0
	else:
		difference[i]=math.sqrt(
							math.pi/2)*(2*logits_list[label]-1)

a=np.mean(difference)
b=a.item()
result1.append(b)
print('Model: {}'.format(online_api_name+'api'+attribute_name),file=data)
print(b,file=data)


store_file_name=online_api_name+'great'+attribute_name+'.npy'
np.save(store_file_name,difference)
#np.save('difference.npy',difference)



