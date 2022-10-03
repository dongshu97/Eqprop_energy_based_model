# coding: utf-8

import os
import os.path
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from scipy import*
from copy import*
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import shutil
from tqdm import tqdm

from Network import*


def train_supervised_ep(net, args, train_loader, epoch):
    net.train()
    net.epoch = epoch + 1

    total_train = torch.zeros(1, device=net.device).squeeze()
    correct_train = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(train_loader):

        # random signed beta: better approximate the gradient
        net.beta = torch.sign(torch.randn(1)) * args.beta

        s = net.initHidden(args, data)

        if net.cuda:
            targets = targets.to(net.device)
            net.beta = net.beta.to(net.device)
            s = [item.to(net.device) for item in s] #no need to put data on the GPU as data is included in s!

        #free phase
        s = net.forward(s)

        seq = s.copy()

        s = net.forward(s, target=targets, beta=net.beta)

        # update and track the weights of the network
        net.updateWeight(s, seq)
        # net.updateWeight(s, seq, beta=net.beta)

        # calculate the training error
        prediction = torch.argmax(seq[0].detach(), dim=1)
        correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        total_train += targets.size(dim=0)

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


def test_supervised_ep(net, args, test_loader):
    '''
    Function to test the network
    '''
    net.eval()

    criterion = nn.MSELoss(reduction = 'sum') #???

    # record total test number
    total_test = torch.zeros(1, device=net.device).squeeze()


    # record supervised test error
    corrects_supervised = torch.zeros(1, device=net.device).squeeze()


    for batch_idx, (data, targets) in enumerate(test_loader):

        s = net.initHidden(args, data)

        if net.cuda:
            targets = targets.to(net.device)
            s = [item.to(net.device) for item in s]

        # record the total test
        total_test += targets.size()[0]

        #free phase
        s = net.forward(s)

        # we note the last layer as s_output
        output = s[0].clone().detach()

        #
        prediction = torch.argmax(output, dim=1)
        corrects_supervised += (prediction == targets).sum().float()

    test_error = 1 - corrects_supervised / total_test
    return test_error


def initDataframe(path, args, net, method='supervised', dataframe_to_init = 'results.csv'):
    '''
    Initialize a dataframe with Pandas so that parameters are saved
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep = ',', index_col = 0)
    else:
        if method == 'supervised':
            columns_header = ['Train_Error', 'Min_Train_Error', 'Test_Error', 'Min_Test_Error']
        elif method == 'unsupervised':
            columns_header = ['Test_Error_av', 'Min_Test_Error_av', 'Test_Error_max', 'Min_Test_Error_max']

        dataframe = pd.DataFrame({}, columns = columns_header)
        dataframe.to_csv(path + prefix + 'results.csv')
    return dataframe


def updateDataframe(BASE_PATH, dataframe, error1, error2):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    data = [error1[-1], min(error1), error2[-1], min(error2)]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)

    try:
        dataframe.to_csv(BASE_PATH + prefix + 'results.csv')
    except PermissionError:
        input("Close the results.csv and press any key.")

    return dataframe


def createPath(args):
    '''
    Create path to save data
    '''
    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    BASE_PATH +=  prefix + 'DATA-0'

    # BASE_PATH += prefix + args.dataset
    #
    # BASE_PATH += prefix + 'method-' + args.method
    #
    # BASE_PATH += prefix + args.action
    #
    # BASE_PATH += prefix + str(len(args.fcLayers)-2) + 'hidden'
    # BASE_PATH += prefix + 'hidNeu' + str(args.layersList[1])
    #
    # BASE_PATH += prefix + 'β-' + str(args.beta)
    # BASE_PATH += prefix + 'dt-' + str(args.dt)
    # BASE_PATH += prefix + 'T-' + str(args.T)
    # BASE_PATH += prefix + 'K-' + str(args.Kmax)
    #
    # BASE_PATH += prefix + 'Clamped-' + str(bool(args.clamped))[0]
    #
    # BASE_PATH += prefix + 'lrW-' + str(args.lrWeights)
    # BASE_PATH += prefix + 'lrB-' + str(args.lrBias)
    #
    # BASE_PATH += prefix + 'BaSize-' + str(args.batchSize)

    BASE_PATH += prefix + datetime.datetime.now().strftime("%Y-%m-%d")


    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    print("len(BASE_PATH)="+str(len(BASE_PATH)))
    filePath = shutil.copy('plotFunction.py', BASE_PATH)

    files = os.listdir(BASE_PATH)

    if 'plotFunction.py' in files:
        files.pop(files.index('plotFunction.py'))

    if not files:
        BASE_PATH = BASE_PATH + prefix + 'S-1'
    else:
        tab = []
        for names in files:
            tab.append(int(names.split('-')[1]))
        BASE_PATH += prefix + 'S-' + str(max(tab)+1)


    os.mkdir(BASE_PATH)
    name = BASE_PATH.split(prefix)[-1]


    return BASE_PATH, name


def saveHyperparameters(args, net, BASE_PATH):
    '''
    Save all hyperparameters in the path provided
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    f = open(BASE_PATH + prefix + 'Hyperparameters.txt', 'w')
    f.write('Classic Equilibrium Propagation - Energy-based settings \n')
    f.write('   Parameters of the simulation \n ')
    f.write('\n')

    for key in args.__dict__:
        f.write(key)
        f.write(': ')
        f.write(str(args.__dict__[key]))
        f.write('\n')

    f.close()
