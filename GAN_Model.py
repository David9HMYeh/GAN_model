#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os, sys, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


torch_dataset = Data.TensorDataset(train, y_target)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # batch size
    shuffle=False,               # random shuffle for training
    num_workers=2,              
)


# In[38]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np

class log_gaussian:

    def __call__(self, x, u, var):

        logli = -0.5*(var.mul(2*np.pi)+1e-5).log() - \ 
                (x-u).pow(2).div(var.mul(2.0)+1e-8)
                

        return logli.sum(1).mean().mul(-1)

class Trainer:
    

    def __init__(self, G, FE, D, Q):

        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q

        self.batch_size = 100

    def _noise_sample(self, dis_c, con_c, noise, bs):

        idx = np.random.randint(2, size=bs)
        c = np.zeros((bs, 2))
        c[range(bs),idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1, 1)
        noise.data.uniform_(-1, 1)
        z0 = torch.cat([noise, dis_c, con_c], 1)
        z = z0.view(-1, 16, 1, 1)
        zzz=[noise, dis_c, con_c]
        return z, idx,zzz

    def train(self):
        count=0

        real_x = torch.FloatTensor(self.batch_size, 1, 32, 32).cuda()
        label = torch.FloatTensor(self.batch_size, 1).cuda()
        dis_c = torch.FloatTensor(self.batch_size, 2).cuda()
        con_c = torch.FloatTensor(self.batch_size, 2).cuda()
        noise = torch.FloatTensor(self.batch_size, 12).cuda()

        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        noise = Variable(noise)

        criterionD = nn.BCELoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=0.00001, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))


        dataloader = loader

        # fixed random variables
        c = np.linspace(-1, 1, 2).reshape(1, -1)
        c = np.repeat(c, 2, 0).reshape(-1, 1)

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])

        idx = np.arange(2).repeat(10)
        one_hot = np.zeros((20, 2))
        one_hot[range(20), idx] = 1
        fix_noise = torch.Tensor(20, 12).uniform_(-0.1, 0.1)


        for epoch in range(150):
            for num_iters, batch_data in enumerate(dataloader, 0):

                # real image
                optimD.zero_grad()

                x, _ = batch_data

                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs, 1)
                dis_c.data.resize_(bs, 2)
                con_c.data.resize_(bs, 2)
                noise.data.resize_(bs, 12)

                real_x.data.copy_(x)

                fe_out1 = self.FE(real_x)

                probs_real = self.D(fe_out1)

                label.data.fill_(1)

                loss_real = criterionD(probs_real, label)

                loss_real.backward()

                # fake part
                z, idx,zzz = self._noise_sample(dis_c, con_c, noise, bs)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = self.Q(fe_out)
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                dis_loss = criterionQ_dis(q_logits, target)
                con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1

                G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()

                if num_iters % 100 == 0:
                    count+=1

                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                    epoch, num_iters, D_loss.data.cpu().numpy(),
                    G_loss.data.cpu().numpy())
                    )
                    


# In[39]:


import torch.nn as nn


class FrontEnd(nn.Module):
    
    def __init__(self):
        super(FrontEnd, self).__init__()

        self.main = nn.Sequential(
          nn.Conv2d(1, 4, 4, 2, 1),
          nn.LeakyReLU(0.1, inplace=True),
          nn.Conv2d(4, 32, 4, 2, 1, bias=False),
          nn.BatchNorm2d(32),
          nn.LeakyReLU(0.1, inplace=True),
          nn.Conv2d(32, 128, 7, bias=False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        output = self.main(x)
        return output


class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        
            )



    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


class Q(nn.Module):

    def __init__(self):
        super(Q, self).__init__()

        self.conv = nn.Conv2d(128, 512, 1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Conv2d(512, 2, 1)
        self.conv_mu = nn.Conv2d(512, 2, 1)
        self.conv_var = nn.Conv2d(512, 2, 1)

    def forward(self, x):

        y = self.conv(x)

        disc_logits = self.conv_disc(y).squeeze()

        u = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()

        return disc_logits, u, var 


class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()

        self.main = nn.Sequential(
          nn.ConvTranspose2d(16, 512, 1, 1, bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(True),
          nn.ConvTranspose2d(512, 256, 1, 1, bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(True),
          nn.ConvTranspose2d(256, 128, 7, 1, bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(True),
          nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(True),
          nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
          nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[ ]:


fe = FrontEnd()
d = D()
q = Q()
g = G()

for i in [fe, d, q, g]:
    i.cuda()
    i.apply(weights_init)

trainer = Trainer(g, fe, d, q)
trainer.train()

