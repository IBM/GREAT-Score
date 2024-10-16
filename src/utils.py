import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dsets

'''
@article{kim2020torchattacks,
  title={Torchattacks: A pytorch repository for adversarial attacks},
  author={Kim, Hoki},
  journal={arXiv preprint arXiv:2010.01950},
  year={2020}
}
'''

# We reference the l2 distance function from torchattack
def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    num_correct = torch.eq(labels.to(device), pre).sum().float().item()
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2,num_correct

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    #print("X size is ", X_exp.size())
    #print("partition size is ", partition, partition.size())
    return X_exp / partition  # 

def tempsigmoid(x):
    nd=3.0
    temp=nd/torch.log(torch.tensor(9.0)) 
    return torch.sigmoid(x/(temp)) 