from time import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom



class INN():
    def __init__(self, ndim_tot, ndim_y, ndim_x,  ndim_z, feature, num_blocks, batch_size, lr, lambd_predict = 1., lambd_latent = 300., lambd_rev = 450., turn=True, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.ndim_tot = ndim_tot
        self.ndim_y =  ndim_y 
        self.ndim_x =  ndim_x 
        self.ndim_z =  ndim_z 
        self.feature = feature
        self.num_blocks = num_blocks
        self.turn = turn
        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, self.feature), nn.ReLU(),
                                 nn.Linear(feature,  c_out))

        nodes = [InputNode(self.ndim_tot, name='input')]

        for k in range(self.num_blocks):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor':subnet_fc, 'clamp':2.0},
                              name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed':k},
                              name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))

        self.model = ReversibleGraphNet(nodes, verbose=False)
        
        # Training parameters
        self.batch_size = batch_size

        self.lr = lr
        l2_reg = 2e-5

        self.y_noise_scale = 1e-1
        self.zeros_noise_scale = 5e-2

        # relative weighting of losses:
        # lambd_predict = 1.
        # lambd_latent = 90.
        # lambd_rev = 100.
        self.lambd_predict = lambd_predict
        self.lambd_latent = lambd_latent
        self.lambd_rev = lambd_rev


        self.pad_x = torch.zeros(self.batch_size, ndim_tot - ndim_x)
        self.pad_yz = torch.zeros(self.batch_size, ndim_tot - ndim_y - ndim_z)

        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr, betas=(0.8, 0.9),
                                     eps=1e-6, weight_decay=l2_reg)

        #def fit(input, target):
        #   return torch.mean((input - target)**2)

        self.loss_backward = self.MMD_multiscale
        self.loss_latent = self.MMD_multiscale
        #self.loss_fit = fit
        self.loss_fit = torch.nn.MSELoss()
        
        for param in trainable_parameters:
            param.data = 0.05*torch.randn_like(param)
        self.model.to(device)
        self.device = device
        return
    
    def train(self, n_epochs, train_loader, val_loader, log_writer=None):
        device = self.device
        for i_epoch in range(n_epochs):
            loss = self.train_epoch(train_loader, i_epoch)
            print(loss)
            if i_epoch % 5 == 0:
                x_samps = torch.cat([x.unsqueeze(1).float() for x,y in val_loader], dim=0)
                y_samps = torch.cat([y.float().unsqueeze(1)  for x,y in val_loader], dim=0)
                N_samp  = x_samps.shape[0]
                
                # If turn == True the validation loss is computed on the [y,z] -> x direction
                if self.turn == True:
                    x_samps, y_samps = y_samps, x_samps
                    y_samps = torch.cat([torch.randn(N_samp, self.ndim_z).to(device),
                                         self.zeros_noise_scale * torch.zeros(N_samp, self.ndim_tot - self.ndim_y - self.ndim_z).to(device), 
                                         y_samps], dim=1)
                    y_samps = y_samps.to(device)

                    rev_x = self.model(y_samps, rev=True)
                    rev_x = torch.nn.Sigmoid()(rev_x)
                    predicted = rev_x[:, 0]
                    ground_truth = x_samps
                    val_loss = torch.nn.MSELoss()(predicted, ground_truth[:,0])
                    print('val loss: {}'.format(val_loss))

                if self.turn == False:
                    padded_x = torch.cat((x_samps, torch.zeros(N_samp, self.ndim_tot - self.ndim_x).to(device)), dim=1)

                    output = self.model(padded_x)[:, -self.ndim_y:]
                    output = torch.nn.Sigmoid()(output)
                    val_loss = torch.nn.MSELoss()(output, y_samps)
                    print('val_loss {}'.format(val_loss.item()))
                if i_epoch == 0:
                    first_loss = val_loss.item()
                if i_epoch > 1000 and first_loss < val_loss.item():
                    break
                if log_writer != None:
                    log_writer.add_scalar('val Loss', val_loss.item(), i_epoch)
            if log_writer != None:
                log_writer.add_scalar('Loss', loss, i_epoch)
        return val_loss.item()
        
    
    def MMD_multiscale(self, x, y):
        device = self.device
        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2.*xx
        dyy = ry.t() + ry - 2.*yy
        dxy = rx.t() + ry - 2.*zz

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        for a in [0.05, 0.2, 0.9]:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

        return torch.mean(XX + YY - 2.*XY)
    
    def train_epoch(self, train_loader, i_epoch=0):
        device = self.device
        self.model.train()

        l_tot = 0
        batch_idx = 0

        # If MMD on x-space is present from the start, the self.model can get stuck.
        # Instead, ramp it up exponetially.  
        #loss_factor = min(1., 2. * 0.002**(1. - (float(i_epoch) / n_epochs)))
        loss_factor = 1
        for x, y in train_loader:
            batch_idx += 1
            
            x, y = x.to(device), y.to(device)

            y = y.unsqueeze(1).float()  

            #Turn
            if self.turn == True:
                y, x = x, y

            y_clean = y.clone()
            self.pad_x = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_tot -
                                                    self.ndim_x, device=device)
            self.pad_yz = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_tot -
                                                     self.ndim_y - self.ndim_z, device=device)
            
            y += self.y_noise_scale * torch.randn(self.batch_size, self.ndim_y, dtype=torch.float, device=device)

            x, y = (torch.cat((x, self.pad_x),  dim=1),
                    torch.cat((torch.randn(self.batch_size, self.ndim_z, device=device), self.pad_yz, y),
                              dim=1))


            self.optimizer.zero_grad()

            # Forward step:

            output = self.model(x)
            output = torch.nn.Sigmoid()(output)

            # Shorten output, and remove gradients wrt y, for latent loss
            y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)

            l = self.lambd_predict * self.loss_fit(output[:, self.ndim_z:], y[:, self.ndim_z:])


            output_block_grad = torch.cat((output[:, :self.ndim_z],

                                           output[:, -self.ndim_y:].data), dim=1)
            l += self.lambd_latent * self.loss_latent(output_block_grad, y_short)
            l_tot += l.data.item()

    #        l.backward()
            l.backward(retain_graph=True)
            # Backward step:
            self.pad_yz = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_tot -
                                                     self.ndim_y - self.ndim_z, device=device)
            y = y_clean + self.y_noise_scale * torch.randn(self.batch_size, self.ndim_y, device=device)

            orig_z_perturbed = (output.data[:, :self.ndim_z] + self.y_noise_scale *
                                torch.randn(self.batch_size, self.ndim_z, device=device))
            y_rev = torch.cat((orig_z_perturbed, self.pad_yz,
                               y), dim=1)
            y_rev_rand = torch.cat((torch.randn(self.batch_size, self.ndim_z, device=device), self.pad_yz,
                                    y), dim=1)

            output_rev = self.model(y_rev, rev=True)
            output_rev_rand = self.model(y_rev_rand, rev=True)
            
            output_rev = (output_rev)
            output_rev_rand = (output_rev_rand)

            l_rev = (
                self.lambd_rev
                * loss_factor
                * self.loss_backward(output_rev_rand[:, :self.ndim_x],
                                x[:, :self.ndim_x])
            )

            l_rev += self.lambd_predict * self.loss_fit(output_rev, x)

            l_tot += l_rev.data.item()
            l_rev.backward()
            for p in self.model.parameters():
                p.grad.data.clamp_(-15.00, 15.00)

            self.optimizer.step()

        return l_tot / batch_idx
