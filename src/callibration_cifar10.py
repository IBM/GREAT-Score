'''
This code is for callibration on cifar10

'''

from ast import Mod, parse
from typing_extensions import Self
from unicodedata import name
from robustbench.data import load_cifar10
import torch
import os 
import os
from torch import nn,optim


from scipy import stats


# we need to define a loss funtion which accepts average GREAT score and average cw attack bound as input, one thing need to note is that if the backward can not be 
# returned, we need to create a 17,5000, list ,first we need to test how to store logits, one thing need to note that we only need to store the right predicted label(how can we slove 0 issue?)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np
device = torch.device("cuda")
from torch.autograd import Variable
import dill
import argparse
import torch.nn.functional as F
import random
import foolbox as fb
import gc
import matplotlib as mpl
import pandas as pd
import seaborn as sns
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from robustbench.data import load_cifar10
from robustbench.data import load_cifar10c
import torch.nn.functional as F



device="cuda"
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    num_correct = torch.eq(labels.to(device), pre).sum().float().item()
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2,num_correct
# We need to modify the temparature scaling , like modifying the temparature smaller than 1



data=open("outputs/callibration.txt",'a')

#path="samples_dcgan.npz"
#path="samples/samples_1000000.npz"
path="samples/samples.npz"

f=np.load(path)
train_X, train_y = f['x'], f['y']
train_X=train_X[0:500]
train_y=train_y[0:500]
#train_X= np.transpose(train_X, (0, 3, 1, 2))
#train_X=train_X.reshape([500, 3, 32, 32])
#train_X=train_X.reshape([500, 3, 32, 32])
f.close()
train_X = train_X.astype('float32')
train_X /= 255
#train_X=train_X-0.5

train_y = train_y.astype('int64')
train_X1=train_X[0:500]
train_y=train_y[0:500]

target_classes=np.zeros(500)



images=torch.from_numpy(train_X1)
labels=torch.from_numpy(train_y)
#images=images[100:400]
#labels=labels[100:400]
images1=images
labels1=labels
unique,count=np.unique(labels,return_counts=True)
data_count=dict(zip(unique,count))
print(data_count)
print(images.shape)



import datetime
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.optim as optim

# we need to print the whole group information 
#model_list=['Engstrom2019Robustness']
model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Sehwag2021Proxy','Augustin2020Adversarial_34_10','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering', 'Sehwag2021Proxy_R18', 'Wu2020Adversarial','Augustin2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Augustin2020Adversarial_34_10','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering', 'Wu2020Adversarial','Augustin2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Sehwag2021Proxy','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering', 'Sehwag2021Proxy_R18', 'Wu2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Augustin2020Adversarial_34_10_extra','Augustin2020Adversarial_34_10','Gowal2020Uncovering', 'Wu2020Adversarial','Augustin2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Sehwag2021Proxy','Augustin2020Adversarial_34_10','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering']
#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Sehwag2021Proxy']
# https://github.com/Harry24k/adversarial-attacks-pytorch
import torchattacks
print("torchattacks %s"%(torchattacks.__version__))
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    #print("X size is ", X_exp.size())
    #print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制


def tempsigmoid(x):
    nd=0.2
    temp=nd/torch.log(torch.tensor(0.04)) 
    return torch.sigmoid(x/(-temp)) 

def tempsoftmax(x, temperature=0.09):
    return np.exp(x/temperature)/sum(np.exp(x/temperature))


'''
for model_name in model_list:
    #model = load_model(model_name, norm='L2').to(device)
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='L2').to(device)
    acc = clean_accuracy(model, images.to(device), labels.to(device))
    print('Model: {}'.format(model_name))
    print('- Standard Acc: {}'.format(acc))
'''
#train_X1=train_X[100:400]
#train_y=train_y[100:400]
X_adv_data=train_X1
Y_data=train_y

h=0

result1=[]



logits_full_list=[]

#model_list=['Rebuffi2021Fixing_70_16_cutmix_extra','Gowal2020Uncovering_extra','Rebuffi2021Fixing_70_16_cutmix_ddpm','Rebuffi2021Fixing_28_10_cutmix_ddpm','Augustin2020Adversarial_34_10_extra','Sehwag2021Proxy','Augustin2020Adversarial_34_10','Rade2021Helper_R18_ddpm','Rebuffi2021Fixing_R18_cutmix_ddpm','Gowal2020Uncovering', 'Sehwag2021Proxy_R18', 'Wu2020Adversarial','Augustin2020Adversarial','Engstrom2019Robustness','Rice2020Overfitting','Rony2019Decoupling','Ding2020MMA']
bound_list=[1.859,1.324,1.943,1.796,1.340,1.392,1.332,1.486,1.413,1.253,1.343,1.369,1.285,1.084,1.097,1.165,1.095]


with torch.no_grad():
    
   
    bound_list=np.array(bound_list)

bound_list1=torch.autograd.Variable(torch.from_numpy(bound_list))


for model_name in model_list:
    '''
    We may consider put all the unpredicted logits into the list
    '''
    logits_local_list=[]
    start1 = datetime.datetime.now()
    print('Model: {}'.format(model_name))
    model = load_model(model_name=model_name, dataset='cifar10', threat_model='L2').to(device)
    #acc = clean_accuracy(model, images.to(device), labels.to(device)) may cause cuda out of memory
    #print(f'Model: {model_name}, CIFAR-10-C accuracy: {acc:.1%}')
    difference=np.zeros(500,dtype=float)

    seq = []
    dif=[]
    j=0
    num_correct= 0
    right=target_classes
    with torch.no_grad():
        for idx in range(len(Y_data)):
                    # load original image
                    
                    # load adversarial image
                    image_adv = np.array(np.expand_dims(X_adv_data[idx], axis=0), dtype=np.float32)
                    #image_adv = np.transpose(image_adv, (0, 3, 1, 2))
                    # load label
                    label = np.array(Y_data[idx], dtype=np.int64)

                    # check bound
                    
                    
                    # transform to torch.tensor
                    data_adv = torch.from_numpy(image_adv).to(device)
                    target = torch.from_numpy(label).to(device)
                    
                    
                    # evluation
                    X, y = Variable(data_adv, requires_grad=True), Variable(target)
                    out = model(X)
                    '''
                    Adjust here to select softmax or sigmoid 
                    '''
                    #out = torch.softmax(out,dim=1)
                    #out=torch.sigmoid(out)
                    #print(out,file=data)
                    logits_local_list.append(out)

                    #print(out[0:10],file=data)
                   
                    #out= torch.softmax(out/0.09, dim=1)
                    out1=out.detach().cpu().numpy()
                    #out1=out
                    #print(out,file=data)
                
                    #out = tempsigmoid(out)
                    #out=torch.sigmoid(out)
                    #print(out,file=data)
                    num_classes = len(out1)
                    #print(out1[0])
                    predicted_label =np.argmax(out1)
                    least_likely_label = np.argmin(out1)
                    start_class = 0 
                    random_class = predicted_label
                    top2_label = np.argsort(out1[0])[-2]
                    #print(top2_label,file=data)
                    #print(out,file=data)
                    new_seq = [least_likely_label, top2_label, predicted_label]
                    #print(new_seq)
                    
                    random_class = random.randint(start_class, start_class + num_classes - 1)
                    new_seq[2] = random_class
                    #true_label = np.argmax(Y_data[idx])
                    true_label =target

                    information = []
                    target_type = 0b0001
                    predicted_label2=np.array(predicted_label)
                    predicted_label2=torch.from_numpy(predicted_label2).to(device)
                    #print(out,file=data)
                    #out=out.data.cpu().numpy()
                    #out=torch.sigmoid(out)
                
                    #print(out,file=data)
                    #out=(1+torch.tanh(out))/2
                    with torch.no_grad():
                        if true_label != predicted_label2:
                        #print(1)
                        #punk=1
                        #seq.append(new_seq[1])
                            difference[j]=0
                            logits=torch.zeros([10])
                            #logits_local_list.append(logits)
                            j=j+1
                        else:
                            if target_type & 0b10000:
                                            for c in range(num_classes):
                                                if c != predicted_label:
                                                    seq.append(c)
                                                    information.append('class'+str(c))
                            else:
                                            if target_type & 0b0100:
                                                # least
                                                seq.append(new_seq[0])
                                                information.append('least')
                                                difference[idx]=math.sqrt(
                                    math.pi/2)*(out[0][predicted_label]-out[0][least_likely_label])
                                                #logits_local_list.append(out)

                                                
                                            if target_type & 0b0001:
                                                # top-2
                                                seq.append(new_seq[1])
                                                #difference[idx]=out[0][predicted_label]-out[0][top2_label]
                                                difference[j]=math.sqrt(
                                    math.pi/2)*(out[0][predicted_label]-out[0][top2_label])
                                                j=j+1
                                                #dif.append(out[0][predicted_label]-out[0][top2_label])
                                                information.append('top2')
                                                #logits_local_list.append(out)
                                                #print(out,file=data)
                                            if target_type & 0b0010:
                                                # random
                                                seq.append(new_seq[2])
                                                difference[idx]=math.sqrt(
                                    math.pi/2)*(out[0][predicted_label]-out[0][random_class])
                                                information.append('random')
                                                #logits_local_list.append(out)
                        
                            #target_classes[idx]=new_seq[1]
                        
                            #out = softmax(out)
                            predicted_label1=np.array(predicted_label)
                            predicted_label1=torch.from_numpy(predicted_label1).to(device)
                            num_correct += torch.eq(predicted_label1, target).sum().float().item()
          
    target_classes=np.array(seq)
    target_classes=torch.from_numpy(target_classes).to(device)
    l2_seq=np.zeros(100,dtype=float)
    acc_seq=np.zeros(100,dtype=float)
    num=np.zeros(100,dtype=float)
    #print(target_classes.shape)



    #print(num_correct,file=data)
 
    #print(target_classes)
    #difference=np.array(dif)
    #print(math.sqrt(math.pi/2)*np.mean(difference[0:int(num_correct)]),file=data)
  
    #print(np.mean(difference),file=data)

    #a=math.sqrt(math.pi/2)*np.mean(difference[0:int(num_correct)])
    a=np.mean(difference)
    
    b=a.item()
    result1.append(b)
    h=h+1
    #print(difference,file=data)
    end1 = datetime.datetime.now()
    #print('time:({} ms)'.format(int((end1-start1).total_seconds())),file=data)
    
    '''
    can we try to ignore the cuda
    '''
    with torch.no_grad():
        logits_array=torch.cat(logits_local_list).cuda()
        logits_full_list.append(logits_array)
    
    
        
    '''
    
    print("- Foolbox")
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    atk = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=1, initial_const=1,
                                        confidence=0, steps=100, stepsize=0.01)
            
    start1 = datetime.datetime.now()
    for idx in range(100):
        
        start = datetime.datetime.now()
        criterion = fb.criteria.TargetedMisclassification(target_classes[idx:idx+1])
        images1=images[idx:idx+1]
        labels1=labels[idx:idx+1]
        _, adv_images, _ = atk(fmodel, images1.to(device), criterion, epsilons=1)
        
        acc = clean_accuracy(model, adv_images, labels1)
        l2,num_correct = l2_distance(model, images1, adv_images, labels1, device=device)
        l2_seq[idx]=l2
        acc_seq[idx]=acc
        num[idx]=num_correct
        end = datetime.datetime.now()
        print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                        int((end-start).total_seconds()*2000)),file=data)
    end1 = datetime.datetime.now()
    sum1=num*l2_seq
    sum2=np.sum(num)
    sum3=np.sum(sum1)
    l2_total=sum3/sum2
    print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(np.mean(acc_seq), l2_total,
                                                        int((end1-start1).total_seconds()*2000)),file=data)'''

# We need to do the callibration process here


'''we put the labels into the module because they are same among all the models'''

class GREATWithTemparature(nn.Module):
    
    #We do not put the model inside the module, we just put temparater into initializer and update it, if this can not work, we may change the great score calculation to a model sequential work
    

    def __init__(self,logits):
         super(GREATWithTemparature, self).__init__()
         self.temperature=nn.Parameter(torch.ones(1)*1.5)
    
    def temparature_scale(self,logits):
        """
        Perform temperature scaling on logits, we may need to adjust the unsqueeze part.
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def tempsoftmax(self, logits):
         return np.exp(logits/self.temperature)/sum(np.exp(logits/self.temperature))
    
    def set_parameter(self,logits,labels,temperature):
        """
        Do we need to pass the labels together with logits? I will set predict process here to compute the difference score.
        """
        result1=[]
        
        self.cuda()
        for h in range(17):
            difference=np.zeros(500,dtype=float)
            seq = []
            dif=[]
            j=0
            num_correct= 0
       
            logits_output=logits[h]
            Y_data=labels
            for idx in range(len(Y_data)):
                label = np.array(Y_data[idx], dtype=np.int64)

                    # check bound
                    
                    
                    # transform to torch.tensor
               
                target = torch.from_numpy(label).to(device)
                out=logits_output[idx]
                #print(out,file=data)
                #out=self.temparature_scale(out)
                #out= torch.sigmoid(out/temperature)
                #print(out,file=data)

                '''
                adjust here to select temperature softmax or sigmoid
                '''
                out= torch.sigmoid(out/temperature)
                #out= torch.softmax(out/temperature, dim=0)
                out1=out.detach().cpu().numpy()
                #out1=out.cpu().numpy()
                #out1=self.temparature_scale(out1)
               
                #out1=self.tempsoftmax(out1)
                #out1=out
                #print(out,file=data)
                
                #out = tempsigmoid(out)
                #out=torch.sigmoid(out)
                #print(out,file=data)
                num_classes = len(out1)
                #print(out1[0])
                predicted_label =np.argmax(out1)
                least_likely_label = np.argmin(out1)
                start_class = 0 
                random_class = predicted_label
                #top2_label = np.argsort(out1[0])[-2]
                top2_label = np.argsort(out1)[-2]
                
                #print(top2_label,file=data)
                #print(out,file=data)
                new_seq = [least_likely_label, top2_label, predicted_label]
                #print(new_seq)
                
                random_class = random.randint(start_class, start_class + num_classes - 1)
                new_seq[2] = random_class
                #true_label = np.argmax(Y_data[idx])
                true_label =target

                information = []
                target_type = 0b0001
                predicted_label2=np.array(predicted_label)
                predicted_label2=torch.from_numpy(predicted_label2).to(device)
                #print(out,file=data)
                #out=out.data.cpu().numpy()
                #out=torch.sigmoid(out)
                
                #print(out,file=data)
                #out=(1+torch.tanh(out))/2
                if true_label != predicted_label2:
                    #print(1)
                    #punk=1
                    #seq.append(new_seq[1])
                    difference[j]=0
                    j=j+1
                else:
                    if target_type & 0b10000:
                                    for c in range(num_classes):
                                        if c != predicted_label:
                                            seq.append(c)
                                            information.append('class'+str(c))
                    else:
                                    if target_type & 0b0100:
                                        # least
                                        seq.append(new_seq[0])
                                        information.append('least')
                                        difference[idx]=math.sqrt(
                            math.pi/2)*(out1[predicted_label]-out1[least_likely_label])

                                        
                                    if target_type & 0b0001:
                                        # top-2
                                        seq.append(new_seq[1])
                                        #difference[idx]=out[0][predicted_label]-out[0][top2_label]
                                        difference[j]=math.sqrt(
                            math.pi/2)*(out1[predicted_label]-out1[top2_label])
                                        j=j+1
                                        #dif.append(out[0][predicted_label]-out[0][top2_label])
                                        information.append('top2')
                                        #print(out,file=data)
                                    if target_type & 0b0010:
                                        # random
                                        seq.append(new_seq[2])
                                        difference[j]=math.sqrt(
                            math.pi/2)*(out1[predicted_label]-out1[random_class])
                                        information.append('random')
                
                    #target_classes[idx]=new_seq[1]
                    
                    #out = softmax(out)
                    predicted_label1=np.array(predicted_label)
                    predicted_label1=torch.from_numpy(predicted_label1).to(device)
                    num_correct += torch.eq(predicted_label1, target).sum().float().item()
    
            target_classes=np.array(seq)
            target_classes=torch.from_numpy(target_classes).to(device)
            l2_seq=np.zeros(100,dtype=float)
            acc_seq=np.zeros(100,dtype=float)
            num=np.zeros(100,dtype=float)
            #print(target_classes.shape)



            #print(num_correct,file=data)
        
            #print(target_classes)
            #difference=np.array(dif)
            #print(math.sqrt(math.pi/2)*np.mean(difference[0:int(num_correct)]),file=data)
            print(difference,file=data)
            #print(np.mean(difference),file=data)

            #a=math.sqrt(math.pi/2)*np.mean(difference[0:int(num_correct)])
            a=np.mean(difference)
            
            b=a.item()
            result1.append(b)
            h=h+1
            print(difference,file=data)
            end1 = datetime.datetime.now()
            print('time:({} ms)'.format(int((end1-start1).total_seconds())),file=data)
        
        #Now we have result1 and CW_attack_bound as two criterias, we need to use MSE error to find the best temparature constant.
        

        mse_criterion=nn.MSELoss().cuda()
        optimizer=optim.LBFGS([self.temperature],lr=0.001,max_iter=50)
        #result2=np.array(result1)
        result2=torch.autograd.Variable(torch.Tensor(result1))
        with torch.no_grad():
            bound_list1=torch.Tensor(bound_list)
        print(result2,file=data)
        
        print(bound_list1,file=data)
        '''
        def eval():
            optimizer.zero_grad()
            loss=mse_criterion(result2,bound_list1)
            loss.backward()
            return loss
        #optimizer.step(eval)
        '''

        return result2



        #Note that we will print the best constant, record it
        
        #print('Optimal temperature: %.3f' % self.temperature.item())

        






'''
When we run callibration, we should print the best temparature, and we test it in temparature scaling file
'''


best_loss= -1
mse_criterion=nn.MSELoss().cuda()


x= np.array([87.20,
85.60,
90.60,
90.00,
86.20,
89.20,
86.40,
86.60,
87.60,
86.40,
88.60,
84.60,
85.20,
82.20,
81.80,
79.20,
77.60])

z=np.array([17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1])

y=np.array(result1)



# we need to change the range
for t in np.arange(0,2,0.0001):  # 1.68,1.71    1.53,1.57 1.5,1.7
 
    Module1=GREATWithTemparature(logits_full_list)
    result=Module1.set_parameter(logits_full_list,Y_data,t)


    
    loss=mse_criterion(result.float(),bound_list1.float())
    compare=torch.le(result.float(),bound_list1.float())
    y=np.array(result)
    

    loss=stats.spearmanr(x, y)[0]

    compare_total=all(compare)
    
    if loss>best_loss and compare_total:
       best_loss=loss
       best_temperature=t
       print(best_temperature)
       print(best_loss)


print("Best score:{:.2f}".format(best_loss))
print("Best parameters:{}".format(best_temperature))
    



# We first search through (0,0.1,0.00005 with the criterion for spearmanr correlation )



data.close()