#Main for the simulation
import os
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision
import torch.optim as optim
import pickle
import datetime
import numpy as np
import platform
import pathlib
import time
from tqdm import tqdm

from Data import *
from Tools import *
from Network import *
from plotFunction import*
from visu import *

parser = argparse.ArgumentParser(description='usupervised EP')
parser.add_argument(
    '--device',
    type=int,
    default=-0,
    help='GPU name to use cuda')
parser.add_argument(
    '--dataset',
    type=str,
    default="mnist",
    help='dataset to be used to train the network : (default = mnist)')
parser.add_argument(
    '--action',
    type=str,
    default="visu",
    help='train or test: (default = supervised_ep, other: test, visu')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    metavar='N',
    help='number of epochs to train (default: 100)')
parser.add_argument(
    '--batchSize',
    type=int,
    default=256,
    help='Batch size (default=256)')
parser.add_argument(
    '--test_batchSize',
    type=int,
    default=256,
    help='Testing batch size (default=256)')
parser.add_argument(
    '--dt',
    type=float,
    default=0.2,
    help='time discretization (default: 0.2)')
parser.add_argument(
    '--T',
    type=int,
    default=60,
    help='number of time steps in the free phase (default: 40) - Let the system relax with oscillators dynamics')
parser.add_argument(
    '--Kmax',
    type=int,
    default=20,
    help='number of time steps in the backward pass (default: 50)')
parser.add_argument(
    '--beta',
    type=float,
    default=0.5,
    metavar='BETA',
    help='nudging parameter (default: 0.5)')
parser.add_argument(
    '--clamped',
    type=int,
    default=1,
    help='Clamped state of the network: crossed input are clamped to avoid divergence (default: True)')
parser.add_argument(
    '--fcLayers',
    nargs='+',
    type=int,
    default=[784, 512, 100],
    help='The structure of fully connected layer')
parser.add_argument(
    '--lr',
    nargs='+',
    type=float,
    default=[0.01, 0.005],
    help='learning rates')
parser.add_argument(
    '--activation_function',
    type=str,
    default='hardsigm',
    help='activation function')
parser.add_argument(
    '--n_class',
    type=int,
    default=10,
    help='the number of class (default = 10)'
)
parser.add_argument(
    '--imWeights',
    type=int,
    default=0,
    help='whether we imshow the weights of synapses'
)
parser.add_argument(
    '--imShape',
    nargs='+',
    type=int,
    default=[28, 28, 16, 16],
    help='decide the size for each imshow of weights'
)
parser.add_argument(
    '--display',
    nargs='+',
    type=int,
    default=[10, 10, 4, 5],
    help='decide the number of neurons whose weights are presented'
)
# input the args
args = parser.parse_args()

# define the two batch sizes
batch_size = args.batchSize
batch_size_test = args.test_batchSize

# if args.dataset == 'digits':
#
#     print('We use the DIGITS Dataset')
#     from sklearn.datasets import load_digits
#     from sklearn.model_selection import train_test_split
#
#     digits = load_digits()
#
#     # TODO make the class_seed of digits dataset
#     x_total, x_class, y_total, y_class = train_test_split(digits.data, digits.target, test_size=0.1, random_state=0,
#                                                           shuffle=True)
#     x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.15, random_state=0, shuffle=True)
#
#     x_class, x_train, x_test = x_class/16, x_train/16, x_test/16  # 0 to 1
#
#     class_data = DigitsDataset(x_class, labels=y_class, target_transforms=ReshapeTransformTarget(10))
#     train_data = DigitsDataset(x_train, labels=y_train, target_transforms=ReshapeTransformTarget(10))
#     test_data = DigitsDataset(x_test, labels=y_test, target_transforms=ReshapeTransformTarget(10))
#
#     len_class = len(x_class[:])
#
#     # dataloaders
#     class_loader = torch.utils.data.DataLoader(class_data, batch_size=batch_size, shuffle=True)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False)


if args.dataset == 'mnist':
    print('We use the MNIST Dataset')
    # Define the Transform
    transforms = [torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

    # Download the MNIST dataset
    if args.action == 'supervised_ep' or args.action == 'test' or args.action == 'visu':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms),
                                               target_transform=ReshapeTransformTarget(10))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchSize, shuffle=True)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.Compose(transforms))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batchSize, shuffle=True)


# define the activation function
if args.activation_function == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))

    def rhop(x):
        return 4*torch.mul(rho(x), 1 -rho(x))

elif args.activation_function == 'hardsigm':
    def rho(x):
        return x.clamp(min=0).clamp(max = 1)

    def rhop(x):
        return (x >= 0) & (x <= 1)

elif args.activation_function == 'tanh':
    def rho(x):
        return torch.tanh(x)

    def rhop(x):
        return 1 - torch.tanh(x)**2


if __name__ == '__main__':

    args.fcLayers.reverse()  # we put in the other side, output first, input last
    args.lr.reverse()
    args.display.reverse()
    args.imShape.reverse()

    # we create the network
    net = MlpEP(args)

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'



    if args.action == 'test':
        print("Testing the model")

        data, target = next(iter(train_loader))

        s = net.initHidden(args, data)

        if net.cuda:
            s = [item.to(net.device) for item in s]
            target = target.to(net.device)

        s, y, h = net.forward(s, tracking=True)

        seq = s.copy()

        s, y1, h1 = net.forward(s, target=target, beta=0.5, tracking=True)
        
        gradW, gradBias = net.computeGradientsEP(s, seq)
        print(gradW[0])

        # update and track the weights of the network
        net.updateWeight(s, seq, args.beta)

        for k in range(len(y)):
            plt.plot(y[k]+y1[k], label=f'Output{k}')
            plt.plot(h[k]+h1[k], '--', label='hidden layer')

        plt.xlabel('Time steps')
        plt.ylabel('Different neuron values')
        plt.axvline(0, ymin=0, ymax=0.8, ls='--', linewidth=1, color='b')
        plt.axvline(args.T-1, ymin=0, ymax=0.8, ls='--', linewidth=1, color='b')
        plt.axvline(args.T + args.Kmax-1, ymin=0, ymax=0.8, ls='--', linewidth=1, color='b')
        plt.title('Dynamics of Equilibrium Propagation')
        plt.show()

    elif args.action == 'supervised_ep':
        print("Training the model with supervised ep")

        BASE_PATH, name = createPath(args)

        saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, args, net, method='supervised')

        # save the initial network
        torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict_0.pt')

        train_error_list = []
        test_error_list = []

        for epoch in tqdm(range(args.epochs)):

            train_error_epoch = train_supervised_ep(net, args, train_loader, epoch)

            test_error_epoch = test_supervised_ep(net, args, test_loader)

            #train_error_list.append(train_error.cpu().item())
            train_error_list.append(train_error_epoch.item())
            test_error_list.append(test_error_epoch.item())

            DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, train_error_list, test_error_list)
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict.pt')

    # we load the pre-trained network
    elif args.action == 'visu':
        BASE_PATH, name = createPath(args)

        net.load_state_dict(torch.load(
            r'C:\..\...\model_state_dict.pt'))  # the path of trained model
        net.eval()

        test_error = test_supervised_ep(net, args, test_loader)

        print('The test error of trained model is:', test_error)

    if args.imWeights:
        # create the imShow dossier
        path_imshow = pathlib.Path(BASE_PATH + prefix + 'imShow')
        path_imshow.mkdir(parents=True, exist_ok=True)
        # for several layers structure
        for i in range(len(args.fcLayers) - 1):
            figName = 'layer' + str(i) + ' weights'
            display = args.display[2 * i:2 * i + 2]
            imShape = args.imShape[2 * i:2 * i + 2]
            weights = net.W[i].weight.data
            if args.device >= 0:
                weights = weights.cpu()
            plot_imshow(weights, args.fcLayers[i], display, imShape, figName, path_imshow, prefix)
            # plot the distribution of weights
            plot_distribution(weights, args.fcLayers[i], 'dis' + figName, path_imshow, prefix)

        # calculate the overlap matrix
        if len(args.fcLayers) > 2:
            overlap = net.W[-1].weight.data
            for j in range(len(args.fcLayers) - 2):
                overlap = torch.mm(net.W[-j - 2].weight.data, overlap)
            if args.device >= 0:
                overlap = overlap.cpu()
            display = args.display[0:2]
            imShape = args.imShape[-2:]
            plot_imshow(overlap, args.fcLayers[0], display, imShape, 'overlap', path_imshow, prefix)

#to run: cf. README in the same folder
