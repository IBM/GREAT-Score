from utils import l2_distance, softmax, tempsigmoid
from scipy import stats
import torch.optim as optim
import torch.nn as nn
from ast import parse
from unicodedata import name
from robustbench.data import load_cifar10
import torch
import os
import os
from autoattack import AutoAttack
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
import numpy as np
import matplotlib.pyplot as plt
import math
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import datetime
import numpy as np
import warnings
import foolbox as fb
warnings.filterwarnings(action='ignore')


device = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
mpl.use('Agg')


def main():
    parser = argparse.ArgumentParser()
    # dataset: cifar-10 default and imagenet ,set -data_dir names according to the dataset
    # different group setting : 1,2,3,4,5 group and comment, follows and output.
    # set a sample directory contains the sample npz or records the command for creating the samples?
    # argument: sample size, activation function,dataset, epsilons, load all the models together, returns final great score and corresponding rank coefficient.
    # set the gpu maybe:
    # set the lower bound mode or autoattack accurcy mode, set a corresponding sample size for attack
    # set proper output format: time for each
    # rewrite a good enough sample random image cw attack individual mode.
    # set different size to process the image. eg:255 -0.5
    # untargeted cw attack and targeted version
    #
    parser.add_argument('--dataset', type=str, default='cifar',
                        help='the dataset want to evaluate on: cifar10 or imagenet')
    parser.add_argument('--activation_function', type=str,
                        default='sigmoid', help='the activation function we used')
    parser.add_argument('--sample_size', type=int,
                        default='500', help='the generated model sample size')
    parser.add_argument('--data_path', type=str,
                        default='samples/samples.npz', help='the samples data path')

    parser.add_argument('--lr', type=float, default=0.01)

    # attack specific settings
    parser.add_argument('--lower_bound_eval_global', action="store_true",
                        help='whether to run cw attack on overall images', default=False)
    parser.add_argument('--lower_bound_eval_local', action="store_true",
                        help='whether to run cw attack on individual images', default=False)
    parser.add_argument('--robust_accuracy', action="store_true",
                        help='whether to run robust accuracy evaluation on over images', default=False)
    parser.add_argument('--ra_n', type=int, default=500,
                        help='the sample size used to run robust accuracy evaluation')
    parser.add_argument('--callibration_method', type=str,default='3',help='the callibration options you want to apply')

    parser.add_argument('--temperature_constant', type=float, default=0.00742)
    args = parser.parse_args()

    # load the generated images and process
    path = args.data_path
    samples = args.sample_size
    temperature_constant=args.temperature_constant
    f = np.load(path)
    train_X, train_y = f['x'], f['y']
    f.close()
    train_X = train_X.astype('float32')
    train_X /= 255
    # train_X=train_X-0.5
    train_y = train_y.astype('int64')
    train_X1 = train_X[0:samples]
    train_y = train_y[0:samples]
    target_classes = np.zeros(samples)
    images = torch.from_numpy(train_X1)
    labels = torch.from_numpy(train_y)
    X_adv_data = train_X1
    Y_data = train_y

    # set a array to store the great score for each model
    great_result1 = []

    # load different models according to the dataset
    if args.dataset.lower() == 'cifar':
        model_list = ['Rebuffi2021Fixing_70_16_cutmix_extra', 'Gowal2020Uncovering_extra', 'Rebuffi2021Fixing_70_16_cutmix_ddpm', 'Rebuffi2021Fixing_28_10_cutmix_ddpm', 'Augustin2020Adversarial_34_10_extra', 'Sehwag2021Proxy', 'Augustin2020Adversarial_34_10',
                      'Rade2021Helper_R18_ddpm', 'Rebuffi2021Fixing_R18_cutmix_ddpm', 'Gowal2020Uncovering', 'Sehwag2021Proxy_R18', 'Wu2020Adversarial', 'Augustin2020Adversarial', 'Engstrom2019Robustness', 'Rice2020Overfitting', 'Rony2019Decoupling', 'Ding2020MMA']
    else:
        #model_list = ['Salman2020Do_50_2', 'Salman2020Do_R50',
                      #'Engstrom2019Robustness', 'Wong2020Fast', 'Salman2020Do_R18']
        model_list=['Salman2020Do_50_2','Salman2020Do_R50','Engstrom2019Robustness','Wong2020Fast','Salman2020Do_R18']

    for model_name in model_list:
       
        print('Model: {}'.format(model_name))
        if args.dataset.lower() == 'cifar':
            model = load_model(model_name=model_name,
                               dataset='cifar10', threat_model='L2').to(device)
        else:
            model = load_model(
                model_name=model_name, dataset='imagenet', threat_model='Linf').to(device)
        start1 = datetime.datetime.now()
        difference = np.zeros(samples, dtype=float)

        seq = []
        dif = []
        j = 0
        num_correct = 0
        right = target_classes
        for idx in range(len(Y_data)):
            # load adversarial image
            image_adv = np.array(np.expand_dims(
                X_adv_data[idx], axis=0), dtype=np.float32)
            #image_adv = np.transpose(image_adv, (0, 3, 1, 2))
            # load label
            label = np.array(Y_data[idx], dtype=np.int64)
            # transform to torch.tensor
            data_adv = torch.from_numpy(image_adv).to(device)
            target = torch.from_numpy(label).to(device)

            # evluation
            X, y = Variable(data_adv, requires_grad=True), Variable(target)
            out = model(X)
            out1 = out.detach().cpu().numpy()
            # print(out)

            #out = tempsigmoid(out)
            # use different activation function according to
            if args.dataset.lower() == 'cifar':
                if args.callibration_method.lower()=='1':
                    out = torch.sigmoid(out/temperature_constant)
                elif args.callibration_method.lower()=='2':
                    out= torch.softmax(out/temperature_constant, dim=1)
                elif args.callibration_method.lower()=='3':
                    out=torch.sigmoid(out)
                    out= torch.softmax(out/temperature_constant, dim=1)
                elif args.callibration_method.lower()=='4':
                    out=torch.softmax(out, dim=1)
                    out=torch.sigmoid(out/temperature_constant)
                  
            else:
                out = softmax(out)
            # print(out)
            num_classes = len(out1)
            predicted_label = np.argmax(out1)
            least_likely_label = np.argmin(out1)
            start_class = 0
            random_class = predicted_label
            top2_label = np.argsort(out1[0])[-2]
            # print(top2_label)
            # print(out)
            new_seq = [least_likely_label, top2_label, predicted_label]
            # print(new_seq)
            random_class = random.randint(
                start_class, start_class + num_classes - 1)
            new_seq[2] = random_class
            #true_label = np.argmax(Y_data[idx])
            true_label = target

            information = []
            target_type = 0b0001
            predicted_label2 = np.array(predicted_label)
            predicted_label2 = torch.from_numpy(predicted_label2).to(device)

            #out = softmax(out)
            # out=(1+torch.tanh(out))/2

            # set the difference to store the local robustness guarantee
            if true_label != predicted_label2:
                # print(1)
                # punk=1
                # seq.append(new_seq[1])
                difference[j] = 0
                j = j+1
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
                        difference[idx] = math.sqrt(
                            math.pi/2)*(out[0][predicted_label]-out[0][least_likely_label])

                    if target_type & 0b0001:
                        # top-2
                        seq.append(new_seq[1])
                        # difference[idx]=out[0][predicted_label]-out[0][top2_label]
                        difference[j] = math.sqrt(
                            math.pi/2)*(out[0][predicted_label]-out[0][top2_label])
                        j = j+1
                        # dif.append(out[0][predicted_label]-out[0][top2_label])
                        information.append('top2')
                        # print(out)
                    if target_type & 0b0010:
                        # random
                        seq.append(new_seq[2])
                        difference[idx] = math.sqrt(
                            math.pi/2)*(out[0][predicted_label]-out[0][random_class])
                        information.append('random')

                # target_classes[idx]=new_seq[1]

                #out = softmax(out)
                predicted_label1 = np.array(predicted_label)
                predicted_label1 = torch.from_numpy(
                    predicted_label1).to(device)
                num_correct += torch.eq(predicted_label1,
                                        target).sum().float().item()

        target_classes = np.array(seq)
        target_classes = torch.from_numpy(target_classes).to(device)
        print(np.mean(difference))
        a = np.mean(difference)
        b = a.item()
        great_result1.append(b)

        end1 = datetime.datetime.now()
        print('time:({} s)'.format(int((end1-start1).total_seconds())))

        if args.robust_accuracy:
            print('AutoAttack on generated images for Model: {}'.format(model_name))
            epsilons = [0.5]
            for epsilons1 in epsilons:
                images1 = images[0:args.ra_n]
                labels1 = labels[0:args.ra_n]
                adversary = AutoAttack(
                    model, norm='L2', eps=epsilons1, version='standard')
                x_adv = adversary.run_standard_evaluation(
                    images1.to(device), labels1.to(device), bs=25)

        if args.lower_bound_eval_global:
            print('CW Attack on generated images for Model: {}'.format(model_name))
            fmodel = fb.PyTorchModel(model, bounds=(0, 1))
            atk = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=7, initial_const=1,
                                        confidence=0, steps=200, stepsize=0.005,abort_early=False)
            
            l2_seq = np.zeros(500, dtype=float)
            acc_seq = np.zeros(500, dtype=float)
            num = np.zeros(500, dtype=float)
            for idx in range(500):
                
                start = datetime.datetime.now()
                criterion = fb.criteria.TargetedMisclassification(target_classes[1*idx:1*idx+1])
                images1=images[1*idx:1*idx+1]
                labels1=labels[1*idx:1*idx+1]
                raw, adv_images, _ = atk(fmodel, images1.to(device), labels1.to(device),epsilons=None)

                acc = clean_accuracy(model, adv_images, labels1)
                
                l2, num_correct = l2_distance(model, images1, adv_images, labels1, device=device)
                l2_seq[idx] = l2
                acc_seq[idx] = acc
                num[idx] = 1-num_correct
                end = datetime.datetime.now()
                print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                                     int((end-start).total_seconds()*1000)))
            end1 = datetime.datetime.now()
            sum1 = num*l2_seq
            sum2 = np.sum(num)
            sum3 = np.sum(sum1)
            l2_total = sum3/sum2
            print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(np.mean(acc_seq), l2_total,
                                                                 int((end1-start1).total_seconds()*1000)))
        if args.lower_bound_eval_local:
            print('CW Attack on individual generated images for Model: {}'.format(model_name))
            print('CW Attack on generated images for Model: {}'.format(model_name))
            fmodel = fb.PyTorchModel(model, bounds=(0, 1))
            atk = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=7, initial_const=1,
                                        confidence=0, steps=200, stepsize=0.005,abort_early=False)
            
            l2_seq = np.zeros(20, dtype=float)
            acc_seq = np.zeros(20, dtype=float)
            num = np.zeros(20, dtype=float)
            for idx in range(20):
                
                start = datetime.datetime.now()
                criterion = fb.criteria.TargetedMisclassification(
                    target_classes[idx:idx+1])
                images1 = images[idx:idx+1]
                labels1 = labels[idx:idx+1]
                raw, adv_images, _ = atk(fmodel, images1.to(device), labels1.to(device),epsilons=None)

                acc = clean_accuracy(model, adv_images, labels1)
                l2, num_correct = l2_distance(model, images1, adv_images, labels1, device=device)
                l2_seq[idx] = l2
                acc_seq[idx] = acc
                num[idx] = 1-num_correct
                end = datetime.datetime.now()
                print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(acc, l2,
                                                                     int((end-start).total_seconds()*1000)))
            end1 = datetime.datetime.now()
            sum1 = num*l2_seq
            sum2 = np.sum(num)
            sum3 = np.sum(sum1)
            l2_total = sum3/sum2
            print('- Robust Acc: {} / L2: {:1.2} ({} ms)'.format(np.mean(acc_seq), l2_total,
                                                                 int((end1-start1).total_seconds()*1000)))
            print(l2_seq)
            print(difference[0:20])
    

    '''
    Below is code for  ranking result
    '''


    print("great score:")
    print(great_result1)
    print("Group ranking result:")
    if args.dataset.lower() == 'cifar':
        # the autoattack accuracy we test on 500 images on the generated data, you can substitue it
        # with others accuracy you recorded in above steps.
        x = np.array([87.20, 85.60, 90.60, 90.00, 86.20, 89.20, 86.40, 86.60,
                     87.60, 86.40, 88.60, 84.60, 85.20, 82.20, 81.80, 79.20, 77.60])
        z= np.array([17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1])
        y = np.array(great_result1)
        print('whole group')
        print(y)
        print(stats.spearmanr(x, y))
        print(stats.spearmanr(x, z))
        print(stats.spearmanr(y, z))
    else:
        x= np.array([
        30.40,
        25.80,
        30.60,
        19.20,
        19.60])
        z = np.array([5, 4, 3, 2, 1])
        y = np.array(great_result1)
        print(stats.spearmanr(x, y))
        print(stats.spearmanr(x, z))
        print(stats.spearmanr(y, z))
       
    
if __name__ == "__main__":
    main()