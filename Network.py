# File defining the network and the oscillators composing the network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import rho, rhop


class MlpEP(nn.Module):

    def __init__(self, args):

        super(MlpEP, self).__init__()

        self.T = args.T
        self.Kmax = args.Kmax
        self.dt = args.dt
        self.beta = torch.tensor(args.beta)
        self.clamped = args.clamped
        self.lr = args.lr

        self.batchSize = args.batchSize

        self.W = nn.ModuleList(None)

        with torch.no_grad():
            for i in range(len(args.fcLayers)-1):
                self.W.extend([nn.Linear(args.fcLayers[i+1], args.fcLayers[i], bias=True)])
                # torch.nn.init.zeros_(self.W[i].bias)

        #put model on GPU is available and asked
        if args.device >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:"+str(args.device))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        self = self.to(device)



    def stepper_c_ep(self, s, target=None, beta=0):
        '''
        stepper function for energy-based dynamics of EP
        '''
        dsdt = []

        #print(rhop(s[0]).float())
        #print(rhop(s[0]).int())

        #
        dsdt.append(-s[0] + (rhop(s[0])*(self.W[0](rho(s[1])))))

        if beta != 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for layer in range(1, len(s)-1):  # start at the first hidden layer and then to the before last hidden layer
            dsdt.append(-s[layer] + rhop(s[layer])*(self.W[layer](rho(s[layer+1])) + torch.mm(rho(s[layer-1]), self.W[layer-1].weight)))

        for (layer, dsdt_item) in enumerate(dsdt):
            s[layer] = s[layer] + self.dt*dsdt_item

            s[0] = s[0].clamp(0, 1)
            if self.clamped:
                s[layer] = s[layer].clamp(0, 1)

        return s

    def stepper_c_ep_vector_beta(self, s, target=None, beta=None):

        dsdt = []

        dsdt.append(-s[0] + (rhop(s[0]) * (self.W[0](rho(s[1])))))

        dsdt[0] = dsdt[0] + beta*(target-s[0])

        for layer in range(1, len(s)-1):  # start at the first hidden layer and then to the before last hidden layer
            dsdt.append(-s[layer] + rhop(s[layer])*(self.W[layer](rho(s[layer+1])) + torch.mm(rho(s[layer-1]), self.W[layer-1].weight)))

        for (layer, dsdt_item) in enumerate(dsdt):
            s[layer] = s[layer] + self.dt*dsdt_item

        if self.clamped:
            s[layer] = s[layer].clamp(0, 1)

        return s

    def forward(self, s, beta=0, target=None, tracking=False):

        T, Kmax = self.T, self.Kmax
        n_track = 4
        q, y = [[] for k in range(n_track)], [[] for k in range(n_track)]

        with torch.no_grad():
            # continuous time EP
            if beta == 0:
                # free phase
                # print('This is free phase', 's[1] size is', s[1].size())
                # print('This is free phase', 's[0] size is', s[0].size())
                for t in range(T):
                    s = self.stepper_c_ep(s)
                    if tracking:
                        for k in range(n_track):
                            # k is the batch number , 2*k is the neuron number
                            y[k].append(s[0][k][2 * k].item())
                            q[k].append(s[1][k][2 * k].item())

            else:
                # nudged phase
                for t in range(Kmax):
                    s = self.stepper_c_ep(s, target=target, beta=beta)
                    if tracking:
                        for k in range(n_track):
                            y[k].append(s[0][k][2 * k].item())
                            q[k].append(s[1][k][2 * k].item())

        if tracking:
            return s, y, q
        else:
            return s

    def computeGradientsEP(self, s, seq):
        '''
        Compute EQ gradient to update the synaptic weight -
        for classic EP! for continuous time dynamics and prototypical
        '''
        batch_size = s[0].size(0)
        # learning rate should be the 1/beta of the BP learning rate
        # in this way the learning rate is correspended with the sign of beta
        coef = 1/(self.beta*batch_size)

        gradW, gradBias = [], []

        with torch.no_grad():
            for layer in range(len(s)-1):
                gradW.append(coef*(torch.mm(torch.transpose(rho(s[layer]), 0, 1), rho(s[layer+1]))-torch.mm(torch.transpose(rho(seq[layer]),0,1),rho(seq[layer+1]))))
                gradBias.append(coef*(rho(s[layer])-rho(seq[layer])).sum(0))

        return gradW, gradBias

    def updateWeight(self, s, seq, epoch=1):
        '''
        Update weights and bias according to EQ algo
        '''

        gradW, gradBias = self.computeGradientsEP(s, seq)

        with torch.no_grad():
            for (layer, param) in enumerate(self.W):
                lr = self.lr[layer]
                param.weight.data += lr * gradW[layer]
                param.bias.data += lr * gradBias[layer]

    def initHidden(self, args, data, testing=False):
        '''
        Init the state of the network
        State if a dict, each layer is state["S_layer"]
        Xdata is the the last element of the dict
        '''
        state = []
        size = data.size(0)
        for layer in range(len(args.fcLayers)-1):
            state.append(torch.zeros(size, args.fcLayers[layer], requires_grad=False))

        state.append(data.float())

        return state










